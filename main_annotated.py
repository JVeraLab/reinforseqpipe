#!/usr/bin/env python3

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from CAI import CAI
import RNA # ViennaRNA
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import math
from itertools import permutations, product
import os
import json
from datetime import datetime
import time

# ==========================================================================================================================================================
# Encoding of Codons
# ==========================================================================================================================================================
class CodonEmbedding:
    """
    Handles the encoding of nucleotides and codons into a numerical vector representation
    for the DQN model's state input.
    """
    def __init__(self):
        # One-hot encoding for each nucleotide (A=0001, T=0010, C=0100, G=1000)
        self.nucleotide_encoding = {
            "A": "0001", "T": "0010", "C": "0100", "G": "1000"
        }
        # The codon vector size is 3 nucleotides * 4 bits/nucleotide = 12 bits.
        self.VECTOR_SIZE = 12
        
        # Standard genetic code mapping codons to amino acids.
        self.genetic_code = {
            "AAA": "K", "AAG": "K", "AAC": "N", "AAT": "N",
            "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
            "AGA": "R", "AGC": "S", "AGG": "R", "AGT": "S",
            "ATA": "I", "ATC": "I", "ATG": "M", "ATT": "I",
            "CAA": "Q", "CAG": "Q", "CAC": "H", "CAT": "H",
            "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
            "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
            "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
            "GAA": "E", "GAG": "E", "GAC": "D", "GAT": "D",
            "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
            "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
            "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
            "TAA": "*", "TAG": "*", "TAC": "Y", "TAT": "Y",
            "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
            "TGA": "*", "TGC": "C", "TGG": "W", "TGT": "C",
            "TTA": "L", "TTC": "F", "TTG": "L", "TTT": "F"
        }
        self.codons = list(self.genetic_code.keys())

    def compute_codon_vector(self, codon: str) -> list[int]:
        """
        Encodes a 3-nucleotide codon into a 12-bit vector by concatenating 
        the 4-bit one-hot encoding of each nucleotide.
        """
        # Concatenate the 4-bit encoding for N1, N2, and N3.
        bin12_str = "".join(self.nucleotide_encoding[nt] for nt in codon)
        # Convert the string of '0's and '1's into a list of integers.
        bin12 = [int(char) for char in bin12_str]
        return bin12 

# ==========================================================================================================================================================
# Verification of Final Sequence
# ==========================================================================================================================================================
class Verification:
    """
    Accessory class to verify that the original and optimized sequences code 
    for the same polypeptide. All other metrics (CAI, MFE, GC) are calculated 
    and reported in the worker function.
    """
    def __init__(self, original_sequence: str, final_sequence: str):
        self.original_sequence = original_sequence
        self.final_sequence = final_sequence

    def compare_dna_sequences(self) -> bool:
        """
        Translates both DNA sequences to check if they result in the same peptide sequence.
        This is the only necessary verification for sequence validity.
        """
        # Seq(dna).translate() automatically handles the genetic code
        original_peptide = Seq(self.original_sequence).translate()
        final_peptide = Seq(self.final_sequence).translate()

        return original_peptide == final_peptide

# ==========================================================================================================================================================
# Environment Definition
# ==========================================================================================================================================================
class MRNAEnv:
    """
    Reinforcement Learning environment for codon optimization via synonymous substitutions.
    The state is the encoded codon sequence.
    Actions are (codon position, new codon).
    Reward is based on the change in a combined CAI/MFE score.
    """
    def __init__(self, original_mrna: list[str], weights: dict, embedding_function, max_steps: int, factor: float, mask_last_n_codons: int):
        
        self.factor = factor
        self.original_mrna = original_mrna.copy() # The mRNA is given as the list of codons.
        self.length_nt = len("".join(self.original_mrna)) # Length in nucleotides.
        self.length_codons = len(self.original_mrna) # Length in codons.
        self.current_mrna = original_mrna.copy() # Current mRNA sequence as a list of codons.
        self.max_steps = max_steps # Max steps is length in nucleotides for 'theory' of mutating each nucleotide once.
        self.steps_taken = 0
        self.mask_last_n_codons = mask_last_n_codons

        # Standard genetic code for internal use (must match CodonEmbedding)
        self.genetics_code = CodonEmbedding().genetic_code
        self.codons = list(self.genetics_code.keys())
        self.synonymous_dict = self._compute_synonymous_groups()
        
        self.prev_cai: float = 0.0 # will be set during reset
        self.prev_mfe: float = 0.0 # will be set during reset

        self.embedding_function = embedding_function
        self.weights = weights

    def _compute_synonymous_groups(self) -> dict[str, list[str]]:
        """Creates a mapping from amino acid to its synonymous codons."""
        syn_dict: dict[str, list[str]] = {}
        for codon, aa in self.genetics_code.items():
            if aa not in syn_dict:
                syn_dict[aa] = []
            syn_dict[aa].append(codon)
        return syn_dict

    def calculate_cai(self, mrna_sequence: str) -> float:
        """Calculates CAI (CAI expects DNA string, T's)."""
        return CAI(mrna_sequence, weights=self.weights)

    def calculate_mfe(self, mrna_sequence: str) -> float:
        """Calculates MFE (ViennaRNA expects RNA string, U's)."""
        return RNA.fold(mrna_sequence.replace("T", "U"))[1]

    def get_initial_state(self) -> tuple[float, float]:
        """Calculates and returns the initial CAI and MFE."""
        dna_seq = "".join(self.original_mrna)
        initial_cai = self.calculate_cai(dna_seq)
        initial_mfe = self.calculate_mfe(dna_seq)
        return initial_cai, initial_mfe

    def reset(self) -> torch.Tensor:
        """Resets the environment to the initial state."""
        self.current_mrna = self.original_mrna.copy()
        self.steps_taken = 0
        
        dna_seq = "".join(self.current_mrna)
        self.prev_cai = self.calculate_cai(dna_seq)
        self.prev_mfe = self.calculate_mfe(dna_seq)

        return self.get_state()

    def get_state(self) -> torch.Tensor:
        """
        Returns the current state as a tensor of shape (seq_length, VECTOR_SIZE) 
        using the codon embedding function.
        """
        state = [self.embedding_function(codon) for codon in self.current_mrna]
        return torch.tensor(state, dtype=torch.float)

    def calculate_mfecaiscore(self, mfe: float, cai: float, l: int, factor: float) -> float:
        """
        Calculates the combined CAI/MFE score, defined as:
        |MFE - L * factor * log10(CAI)|, where L is the number of codons.
        This is the value the RL agent is trying to minimize.
        """
        # Ensure log10(cai) is defined
        if cai <= 0:
            # Handle non-positive CAI by returning a very large positive number
            # to penalize the state heavily.
            return 1e9 
        return abs(mfe - l * factor * math.log10(cai))

    def step(self, position: int, new_codon: str) -> tuple[torch.Tensor, float, bool, dict]:
        """
        Applies the action (synonymous substitution) to the environment.
        
        Returns: next_state, reward, done, info
        """
        old_codon = self.current_mrna[position]
        old_aa = self.genetics_code[old_codon]
        new_aa = self.genetics_code[new_codon]

        # 1. Check for illegal (non-synonymous) move
        if old_aa != new_aa:
            # Illegal move: non-synonymous substitution attempted (should be prevented by the mask, but checked here for safety).
            reward = -1.0 # Heavy penalty
            done = True # Terminate episode
            info = {"error": "Non-synonymous substitution attempted"}
            return self.get_state(), reward, done, info

        # 2. Update state and step counter
        self.current_mrna[position] = new_codon
        self.steps_taken += 1

        # 3. Calculate metrics for the new state
        dna_seq = "".join(self.current_mrna)
        new_cai = self.calculate_cai(dna_seq)
        new_mfe = self.calculate_mfe(dna_seq)

        # 4. Calculate reward (change in the MFExCAI score)
        # Reward is defined as idfferense in score: new_score - old_score
        # Maximizing the score means maximizing the reward.
        
        old_score = self.calculate_mfecaiscore(self.prev_mfe, self.prev_cai, self.length_codons, self.factor)
        new_score = self.calculate_mfecaiscore(new_mfe, new_cai, self.length_codons, self.factor)
        reward = new_score - old_score # Maximize the decrease in score.
        
        # 5. Update previous values
        self.prev_cai = new_cai
        self.prev_mfe = new_mfe

        # 6. Check for termination
        done = self.steps_taken >= self.max_steps

        return self.get_state(), reward, done, info

    def get_allowed_actions(self, position: int) -> list[str]:
        """
        For a given codon position, returns the list of allowed synonymous codons.
        The last `mask_last_n_codons` positions are fixed (no change allowed).
        """
        if position >= self.length_codons - self.mask_last_n_codons:
            # For masked codons (like the stop codon or C-terminal region), no change is allowed.
            return [self.current_mrna[position]]
        
        current_codon = self.current_mrna[position]
        aa = self.genetics_code[current_codon]
        syn_codons = self.synonymous_dict[aa]
        
        # Allowed actions are any synonymous codons *other than* the current one.
        # This is because the problem is framed as substitutions.
        allowed = [codon for codon in syn_codons if codon != current_codon]
        
        # If no synonymous alternatives exist (e.g., for ATG/Trp), allow the no-op (current codon)
        # to ensure at least one action is available, which is then masked in `compute_allowed_mask`.
        if not allowed:
             allowed.append(current_codon) 
             
        return allowed

# ==========================================================================================================================================================
# Helper Function: Compute Allowed Mask
# ==========================================================================================================================================================
def compute_allowed_mask(env: MRNAEnv) -> torch.Tensor:
    """
    Computes a boolean mask for legal actions based on the current environment state.
    
    The mask is flattened to shape (seq_length * num_codons,).
    An action is legal if it corresponds to a synonymous codon change (or no-op for fixed codons).
    """
    seq_length = len(env.current_mrna)
    num_codons = len(env.codons)
    
    # Initialize mask to all False
    allowed_mask = torch.zeros(seq_length, num_codons, dtype=torch.bool)
    
    for i in range(seq_length):
        allowed_codons = env.get_allowed_actions(i)
        # Note: get_allowed_actions returns a list of *substitutions*, 
        # but for positions that cannot be changed (masked or no alternatives), 
        # it returns the current codon. This distinction is crucial for RL action space.
        
        # If no substitutions are available (e.g., Met/Trp, or masked codon), 
        # the no-op (choosing the current codon) is the only legal action.
        current_codon = env.current_mrna[i]
        
        # If the only 'allowed' codon is the current one, it means no substitution is legal.
        is_fixed = (len(allowed_codons) == 1 and allowed_codons[0] == current_codon)
        
        for j in range(num_codons):
            codon = env.codons[j]
            # If the codon is one of the allowed synonymous substitutions
            if codon in allowed_codons:
                allowed_mask[i, j] = True
            # If the position is fixed, the only legal action is the no-op (choosing the current codon again).
            elif is_fixed and codon == current_codon:
                allowed_mask[i, j] = True
            
    return allowed_mask.view(-1)

# ==========================================================================================================================================================
# Q-Network (DQN Model)
# ==========================================================================================================================================================
class DQNModel(nn.Module):
    """
    Deep Q-Network model using an LSTM to process the sequential codon state.
    
    Input: (batch_size, seq_length, input_size) -> Encoded codon sequence.
    Output: (batch_size, seq_length * num_codons) -> Q-values for all (position, codon) actions.
    """
    def __init__(self, input_size: int, hidden_size: int, seq_length: int, num_codons: int):
        super().__init__()
        self.seq_length = seq_length
        self.num_codons = num_codons
        
        # LSTM layer to capture sequential context of codons.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Fully connected layer to output Q-values for each possible codon substitution 
        # at each position. The output size is num_codons for each position.
        self.fc = nn.Linear(hidden_size, num_codons)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(state)  # shape: (batch_size, seq_length, hidden_size)
        
        # Apply FC layer to get Q-values for each position.
        q_values = self.fc(lstm_out)    # shape: (batch_size, seq_length, num_codons)
        
        # Flatten the output to represent the Q-value for every possible (position, codon) action.
        q_values = q_values.view(state.size(0), self.seq_length * self.num_codons)
        return q_values

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
        # This mask ensures only synonymous substitutions are chosen.
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
        # Gradient clipping might be added here, but preserving original logic.
        self.optimizer.step()
        
    def update_target_network(self):
        """Updates the target network weights to match the policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def action_to_components(self, action: int) -> tuple[int, int]:
        """
        Converts the flattened action index into (codon position, codon index).
        """
        position = action // self.num_codons
        codon_index = action % self.num_codons
        return position, codon_index

# ==========================================================================================================================================================
# Training Loop
# ==========================================================================================================================================================
def train_agent_dqn(original_dna: str, device: torch.device, weights: dict, num_episodes: int, 
                    max_steps: int, factor: float, mask_last_n_codons: int) -> tuple:
    """
    Main training function for the DQN agent.
    
    Returns: A tuple containing various initial/final metrics and episode data lists.
    """
    # Split DNA sequence into a list of codons.
    original_mrna = [original_dna[i:i+3] for i in range(0, len(original_dna), 3)]
    
    # Initialize embedding helper
    computing_codon = CodonEmbedding() 
    
    # Initialize the Environment
    env = MRNAEnv(
        original_mrna,
        weights,
        computing_codon.compute_codon_vector,
        max_steps=max_steps, # Max steps is length in nucleotides.
        factor=factor,
        mask_last_n_codons=mask_last_n_codons 
    )
    
    initial_cai, initial_mfe = env.get_initial_state()
    seq_length = len(original_mrna)
    
    # Hyperparameters (preserved from hyperparameters optimziation for specific usage case)
    input_size = 12 
    num_codons = 64
    lr = 1e-2
    gamma=0.95
    target_update = 30
    batch_size = 64
    hidden_size = 128
    epsilon_decay = 1.5 * len(original_dna)
    
    # Initialize the Agent
    agent = DQNAgent(input_size, hidden_size, seq_length, num_codons, device, lr, gamma, 
                     target_update, batch_size, epsilon_decay, epsilon_start=1.0, 
                     epsilon_final=0.05, buffer_capacity=10000)
    
    # Tracking variables
    total_steps = 0
    best_cai = initial_cai
    best_mfe = initial_mfe
    best_score = env.calculate_mfecaiscore(initial_mfe, initial_cai, seq_length, factor)
    best_sequence = "".join(original_mrna)
    best_sequences: list[str] = []
    
    # Data logging
    episode_number: list[int] = []
    episode_cai: list[float] = []
    episode_mfe: list[float] = []
    episode_score: list[float] = []

    for episode in range(1, num_episodes + 1):
        # Reset environment
        state_tensor = env.reset() 
        state = state_tensor.unsqueeze(0).to(device) # shape: (1, seq_length, vector_size)
        episode_reward = 0.0
        
        for t in range(max_steps):
            # 1. Select action
            action = agent.select_action(state, env)
            position, codon_idx = agent.action_to_components(action)
            chosen_codon = env.codons[codon_idx]
            
            # 2. Take the step in the environment
            next_state_tensor, reward, done, info = env.step(position, chosen_codon)
            episode_reward += reward
            next_state = next_state_tensor.unsqueeze(0).to(device)
            
            # 3. Compute allowed mask for the *next* state 
            # (required for target Q-value calculation in update_policy)
            allowed_mask = compute_allowed_mask(env)
            
            # 4. Store transition in replay memory
            agent.memory.push(state, action, reward, next_state, allowed_mask.cpu(), done)
            
            state = next_state
            total_steps += 1
            
            # 5. Perform optimization
            agent.update_policy()
            
            if done:
                if "error" in info:
                    print(f"Episode {episode} terminated early: {info['error']}")
                break

        # 6. Periodically update the target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # 7. Log episode metrics
        current_sequence = "".join(env.current_mrna)
        current_cai = env.calculate_cai(current_sequence)
        current_mfe = env.calculate_mfe(current_sequence)
        current_score = env.calculate_mfecaiscore(current_mfe, current_cai, seq_length, factor)
        
        episode_number.append(episode)
        episode_cai.append(current_cai)
        episode_mfe.append(current_mfe)
        episode_score.append(current_score)
        
        # 8. Update best sequence
        # The goal is to MINIMIZE the score, so we check for < best_score.
        if current_score < best_score:
            best_cai = current_cai
            best_mfe = current_mfe
            best_score = current_score
            best_sequence = current_sequence
            best_sequences.append(best_sequence) # Log all best sequences (optional: only store the final one)
            
    # Final calculation for index (CAI * |MFE|) for comparison
    original_idx = initial_cai * abs(initial_mfe)
    best_idx = best_cai * abs(best_mfe)
            
    return (initial_cai, initial_mfe, env.calculate_gc_content(original_dna), # Original metrics
            best_sequence, best_cai, best_mfe, env.calculate_gc_content(best_sequence), # Best metrics
            episode_number, episode_cai, episode_mfe, episode_score, original_idx, best_idx)

# ==========================================================================================================================================================
# Assembling of Candidate Peptides
# ==========================================================================================================================================================
class Assemble:
    """
    Handles the assembly of candidate peptide sequences and their back-translation 
    into DNA using high-usage codons.
    """
    def __init__(self, codon_usage_file: str):
        # Peptide fragments to be permuted
        self.peptides_list = ['YLEPGPVTA', 'EAAGIGILTV', 'EVDPIGHLY', 'SLYNTVATL', 'ILKEPVHGV']
        # Linker sequence
        self.linker = ['GGGGS']
        # Marker tag sequence
        self.myctag_aa = 'EQKLISEEDL'
        self.myctag_dna = "GAACAAAAACTCATCTCAGAAGAGGATCTG" # DNA of EQKLISEEDL (if the original code meant a DNA tag)
        
        # Codon usage (from the JSON file)
        with open(codon_usage_file, 'r') as file:
            self.codon_usage: dict[str, float] = json.load(file)
            
        # Codon table for back-translation
        self.genetic_code = CodonEmbedding().genetic_code
        self.aa_to_codons: dict[str, list[str]] = CodonEmbedding()._compute_synonymous_groups()
            
        self.stopcodon = 'TAA'
        self.startcodon = 'ATG'
        
    def construct_peptide_sequences(self) -> list[str]:
        """
        Generates all possible permutations of peptide sequences with all combinations of linkers,
        and appends the marker tag (`EQKLISEEDL`).
        """
        peptide_permutations = list(permutations(self.peptides_list))
        sequences: list[str] = []

        for perm in peptide_permutations:
            # Handle single peptide sequences (though linkers are used between them)
            if len(perm) == 1:
                sequences.append(perm[0] + self.linker[0] + self.myctag_aa)
                continue

            # Assuming only one linker type for now, as per the original logic
            linker_combinations = product(self.linker, repeat=len(perm) - 1)
            for linkers in linker_combinations:
                # Start with the first peptide
                base_sequence = perm[0]
                # Add each linker and next peptide
                for i, linker in enumerate(linkers):
                    base_sequence += linker + perm[i + 1]
                # Append the final linker and the marker tag
                sequences.append(base_sequence + self.linker[0] + self.myctag_aa)

        return sequences
    
    def backtranslate_peptide(self) -> tuple[list[str], int]:
        """
        Back-translates each assembled peptide sequence into a DNA sequence 
        using the codon with the highest usage, and adds start/stop codons.
        """
        final_constructs: list[str] = []
        
        for peptide in self.construct_peptide_sequences():
            dna_sequence: list[str] = []
            for aa in peptide:
                if aa in self.aa_to_codons:
                    codons = self.aa_to_codons[aa]
                    # Select the codon with the highest usage from the codon usage dictionary
                    best_codon = max(codons, key=lambda codon: self.codon_usage.get(codon, 0))
                    dna_sequence.append(best_codon)
                else:
                    raise ValueError(f"Unknown amino acid: {aa}")
                
            # Add start and stop codons
            full_dna_sequence = self.startcodon + "".join(dna_sequence) + self.stopcodon
            final_constructs.append(full_dna_sequence)

        # The original code calculates the length of the tag+stop in codons for the mask.
        # However, the mask is set to 0 in main, so this return value is unused there. 
        # Returning 0 as it's the value used in the main function.
        return final_constructs, 0


# ==========================================================================================================================================================
# Worker and Multiprocessing
# ==========================================================================================================================================================
def worker(sequence: str, device: torch.device, weights: dict, num_episodes: int, 
           max_steps: int, index: int, factor: float, mask_last_n_codons: int) -> tuple[int, dict] | None:
    """
    Worker function executed in a separate process to run the DQN training loop 
    for a single sequence and gather the results.
    The scoring function is given as MFECAIlambda score: |MFE - L * factor * log10(CAI)|
    """
    print(f"Worker {index} starting.")
    start_time = time.perf_counter()
    
    # Run the training loop
    (original_dna_cai, original_dna_mfe, original_dna_gc_content,
     best_sequence, best_sequence_cai, best_sequence_mfe, best_sequence_gc_content,
     episode_number, episode_cai, episode_mfe, episode_score, original_idx, best_idx) = train_agent_dqn(
        sequence, device, weights, num_episodes, max_steps, factor, mask_last_n_codons)
        
    end_time = time.perf_counter()
    time_taken = end_time - start_time

    # Verification
    verification = Verification(sequence, best_sequence)
    verified = verification.compare_dna_sequences()
    
    # Calculate MFECAIlambda score (The value the agent minimizes)
    seq_length_codons = len(sequence) // 3
    orig_mfe_cai_score = abs(original_dna_mfe - factor * seq_length_codons * math.log10(original_dna_cai))
    final_mfe_cai_score = abs(best_sequence_mfe - factor * seq_length_codons * math.log10(best_sequence_cai))

    # Compile results
    results_row = {
        "Original Sequence": sequence, 
        "Final Sequence": best_sequence, 
        "Episode List": list(episode_number),
        "CAI List": list(episode_cai), 
        "MFE List": list(episode_mfe),
        "Score List": list(episode_score), 
        "Original CAI": round(original_dna_cai, 4), 
        "Original MFE": round(original_dna_mfe, 4),
        "Original GC content": round(original_dna_gc_content, 4),
        "Original Index (CAI*|MFE|)": round(original_idx, 4), # The score is not used for optimization
        "Final CAI": round(best_sequence_cai, 4), 
        "Final MFE": round(best_sequence_mfe, 4), 
        "Final GC content": round(best_sequence_gc_content, 4),
        "Final Index (CAI*|MFE|)": round(best_idx, 4),
        "Original MFExCAI score": round(orig_mfe_cai_score, 4),
        "Final MFExCAI score": round(final_mfe_cai_score, 4),
        "The gain of CAI score:": round(best_sequence_cai - original_dna_cai, 4),
        "The gain of MFE score:": round(original_dna_mfe - best_sequence_mfe, 4), # MFE is negative, so this measures reduction in folding energy (making it stronger)
        "The gain of Index score": round(best_idx - original_idx, 4),
        "Peptide_compatibility": bool(verified),
        "Time Taken (s)": round(time_taken, 4)
    }
    
    print(f"Worker {index} finished.")
    return (index, results_row)


def main(weights_path: str, factor: float, episodes: int, sequences_path: str, batch_size:int, generation: bool = True):
    """
    Main execution function orchestrating sequence assembly, environment setup, 
    and parallel processing of the DQN optimization.
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set start method for multiprocessing compatibility with CUDA
    mp.set_start_method('spawn', force=True) 

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
    # Setup output folder
    folder_name = f'results_optimization'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 1. Sequence Generation/Loading
    sequences: list[str]
    # mask_last_n_codons: int # The number of C-terminal codons to fix (not allow mutation)
    sequences_path = 'sequences.json'
    if generation:
        print("Assembling sequences...")
        assembling = Assemble(weights_path)
        sequences, mask_last_n_codons = assembling.backtranslate_peptide()
    else: 
        with open('sequences.json', 'r') as file:
            sequences = json.load(file)

    # 2. Load CAI weights
    with open(weights_path, 'r') as file:
        weights: dict = json.load(file)

    if not isinstance(weights, dict):
        raise ValueError("weights should be a dictionary")
        
    # The original code sets mask_last_n_codons to 0 right before the loop, 
    # overriding the value from backtranslate_peptide.
    
    # 3. Parallel Processing
    df_results = pd.DataFrame()
    batch_size = batch_size # Number of sequences to process in parallel
    num_episodes = episodes 
    
    i = 1
    
    # Iterate through sequences in batches
    for batch_start in range(0, len(sequences), batch_size):
        print(f"Processing batch number {i}")
        batch = sequences[batch_start:batch_start + batch_size]
        
        # Determine max_steps for the current batch (max length in the batch)
        # Using the length of the sequence itself for max_steps per the original logic
        # and len(sequence) for each individual worker.
        
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            # max_steps argument passed to worker is len(sequence)
            futures = [
                executor.submit(worker, sequence, device, weights, num_episodes, 
                                len(sequence), batch_start + j, factor, mask_last_n_codons)
                for j, sequence in enumerate(batch)
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is not None and result[1] is not None:
                    # Append result to DataFrame
                    # Using df_results._append for pandas 2.0+ compatibility
                    df_results = df_results._append(result[1], ignore_index=True) 
                else:
                    print(f"Received None result for index {result[0]}")

        print(f"Finished batch number {i}")
        i += 1
        
        # Clear CUDA cache between batches to free up memory
        torch.cuda.empty_cache()
        time.sleep(2) 
        
    # 4. Save Results
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{folder_name}/results_for_run_dated_{formatted_time}_{factor}.csv"
    df_results.to_csv(filename, index=True)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    # Configuration
    weights_pathes = 'codon_usage_dict_linear_design.json'
    episodes_per_run = 20 # num_episodes argument
    lambda_parameter = 4 # factor argument
    batch_size = 50 # Number of sequences to process in parallel
    main(weights_pathes, lambda_parameter, episodes_per_run, batch_size, generation=True)