# core/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .model import DQNModel
from .env import MRNAEnv, compute_allowed_mask

# ==========================================================================================================================================================
# Experience Replay Buffer
# ==========================================================================================================================================================
class ReplayBuffer:
    """
    Standard Replay Buffer for storing and sampling experience tuples.
    A transition is (state, action, reward, next_state, allowed_mask, done).
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[tuple] = []
        self.position = 0
        
    def push(self, state: torch.Tensor, action: int, reward: float, 
             next_state: torch.Tensor, allowed_mask: torch.Tensor, done: bool):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # type: ignore
        self.buffer[self.position] = (state, action, reward, next_state, allowed_mask, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> list[tuple]:
        """Randomly samples a batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

# ==========================================================================================================================================================
# DQN Agent
# ==========================================================================================================================================================
class DQNAgent:
    """
    DQN Agent implementing the epsilon-greedy policy and the Q-network update logic.
    """
    def __init__(self, input_size: int, hidden_size: int, seq_length: int, num_codons: int, 
                 device: torch.device, lr: float, gamma: float, target_update: int, 
                 batch_size: int, epsilon_decay: float, epsilon_start: float=1.0, 
                 epsilon_final: float=0.05, buffer_capacity: int=10000):
        
        self.seq_length = seq_length
        self.num_codons = num_codons
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.device = device
        
        # Initialize policy and target networks
        self.policy_net = DQNModel(input_size, hidden_size, seq_length, num_codons).to(self.device)
        self.target_net = DQNModel(input_size, hidden_size, seq_length, num_codons).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        
    def get_epsilon(self) -> float:
        """Calculates the current epsilon value using exponential decay."""
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                np.exp(-1. * self.steps_done / self.epsilon_decay)
        
    def select_action(self, state: torch.Tensor, env: MRNAEnv) -> int:
        """
        Implements the epsilon-greedy policy to select an action (an integer index). 
        
        Action index is flattened: `action = position * num_codons + codon_index`.
        """
        epsilon = self.get_epsilon()
        self.steps_done += 1

        # 1. Compute allowed action mask for the current state.
        allowed_mask = compute_allowed_mask(env).to(self.device)
        
        # 2. Epsilon-greedy exploration
        if random.random() < epsilon:
            # Explore: choose a random allowed action
            allowed_indices = torch.nonzero(allowed_mask).squeeze()
            if allowed_indices.dim() == 0:
                # Handle the case where only one action (no-op) is allowed
                chosen = allowed_indices.item()
            else:
                chosen = random.choice(allowed_indices).item()
            return chosen

        # 3. Exploit: choose the best allowed action according to Q-values.
        with torch.no_grad():
            q_values = self.policy_net(state)  # shape: (1, seq_length*num_codons)
            q_values = q_values.squeeze()
            
            # Mask out illegal actions by assigning a very low value, effectively ignoring them.
            q_values[~allowed_mask] = -1e9
            
            chosen = int(torch.argmax(q_values).item())
            return chosen

    def update_policy(self):
        """Performs one step of optimization on the policy network using sampled transitions."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (convert list of tuples to tuple of lists)
        batch = list(zip(*transitions))
        
        # Convert to Tensors
        # state_batch shape: (batch_size, seq_length, input_size)
        state_batch = torch.cat(batch[0]).to(self.device) 
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device) 
        reward_batch = torch.tensor(batch[2], dtype=torch.float, device=self.device) 
        next_state_batch = torch.cat(batch[3]).to(self.device)
        allowed_mask_batch = torch.stack(batch[4]).to(self.device) # Mask for the next state
        done_batch = torch.tensor(batch[5], dtype=torch.float, device=self.device) 

        # 1. Compute Q(s, a) for the taken actions (current state, policy net)
        q_values = self.policy_net(state_batch) 
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 2. Compute max Q(s_{t+1}, a') using the target network (max over allowed actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch) 
            
            # Mask illegal actions in the next state's Q-values
            next_q_values[~allowed_mask_batch] = -1e9
            next_state_values = next_q_values.max(dim=1)[0]
            
            # Compute the target Q values: R + gamma * max_a' Q(s_{t+1}, a')
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        # 3. Compute the loss (Mean Squared Error)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 4. Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        """Updates the target network weights to match the policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def action_to_components(self, action: int) -> tuple[int, int]:
        """
        Converts the flattened action index into (codon position, codon index).
        """
        # Assumes the action space is flattened by (position * num_codons + codon_index)
        position = action // self.num_codons
        codon_index = action % self.num_codons
        return position, codon_index