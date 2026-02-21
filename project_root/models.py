import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from environment import MRNAEnv, compute_allowed_mask



class DQNModel(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_codons):
        super(DQNModel, self).__init__()
        self.seq_length = seq_length
        self.num_codons = num_codons
        # Use an LSTM to incorporate sequence context.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # For each position, output Q-values for each codon (of size num_codons).
        self.fc = nn.Linear(hidden_size, num_codons)
        
    def forward(self, state):
        # state shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(state)  # shape: (batch_size, seq_length, hidden_size)
        q_values = self.fc(lstm_out)    # shape: (batch_size, seq_length, num_codons)
        # Flatten to (batch_size, seq_length * num_codons)
        q_values = q_values.view(state.size(0), self.seq_length * self.num_codons)
        return q_values



class ReplayBuffer:
    """
    Global update of the model state. On practice the update is not used.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, allowed_mask, done):
        # Tuple structure: (state, action, reward, next_state, allowed_mask, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, allowed_mask, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_size, hidden_size, seq_length, num_codons, device, lr,
                 gamma, target_update, batch_size, epsilon_decay, epsilon_start=1.0, epsilon_final=0.05,
                 buffer_capacity=10000):
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
        self.policy_net = DQNModel(input_size, hidden_size, seq_length, num_codons).to(self.device)
        self.target_net = DQNModel(input_size, hidden_size, seq_length, num_codons).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_capacity)
        
    def get_epsilon(self):
        # Exponential decay of epsilon
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
               np.exp(-1. * self.steps_done / self.epsilon_decay)
        
    def select_action(self, state, env):
        """
        state: tensor of shape (1, seq_length, input_size)
        Returns:
            action (int) corresponding to a (position, codon) pair.
        """
        epsilon = self.get_epsilon()
        self.steps_done += 1

        # Compute allowed action mask for the current state using env.
        allowed_mask = torch.zeros(self.seq_length, self.num_codons, dtype=torch.bool)
        for i in range(self.seq_length):
            allowed_codons = env.get_allowed_actions(i)
            if not allowed_codons:
                allowed_codons = [env.current_mrna[i]]
            for j in range(self.num_codons):
                if env.codons[j] in allowed_codons:
                    allowed_mask[i, j] = True
        allowed_mask = allowed_mask.view(-1)

        # With probability epsilon, choose a random allowed action.
        if random.random() < epsilon:
            allowed_indices = torch.nonzero(allowed_mask).squeeze().tolist()
            if isinstance(allowed_indices, int):
                chosen = allowed_indices
            else:
                chosen = random.choice(allowed_indices)
            return chosen

        with torch.no_grad():
            q_values = self.policy_net(state)  # shape: (1, seq_length*num_codons)
            q_values = q_values.cpu().squeeze()
            # Mask out mutatiosn leading to another amino acid
            q_values[~allowed_mask] = -1e9
            chosen = int(torch.argmax(q_values).item())
            return chosen

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch; each element is a tuple: (state, action, reward, next_state, allowed_mask, done)
        batch = list(zip(*transitions))
        state_batch = torch.cat(batch[0]).to(self.device)          # shape: (batch, seq_length, input_size)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device)  # shape: (batch,)
        reward_batch = torch.tensor(batch[2], dtype=torch.float, device=self.device) # shape: (batch,)
        next_state_batch = torch.cat(batch[3]).to(self.device)
        allowed_mask_batch = torch.stack(batch[4]).to(self.device)   # shape: (batch, seq_length*num_codons)
        done_batch = torch.tensor(batch[5], dtype=torch.float, device=self.device)   # shape: (batch,)

        # Compute Q(s, a) for the taken actions.
        q_values = self.policy_net(state_batch)              # shape: (batch, seq_length*num_codons)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute V(s_{t+1}) using the target network.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)  # shape: (batch, seq_length*num_codons)
            # Mask illegal actions using the stored allowed mask.
            next_q_values[~allowed_mask_batch] = -1e9
            next_state_values = next_q_values.max(dim=1)[0]
            # Compute the target Q values.
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        # Compute the loss (MSE).
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def action_to_components(self, action):
        """
        Convert the discrete action index into (position, codon).
        """
        position = action // self.num_codons
        codon_index = action % self.num_codons
        return position, codon_index

