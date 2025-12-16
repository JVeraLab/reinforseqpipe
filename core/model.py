# core/model.py

import torch
import torch.nn as nn

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
        # at each position.
        self.fc = nn.Linear(hidden_size, num_codons)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(state)  # shape: (batch_size, seq_length, hidden_size)
        
        # Apply FC layer to get Q-values for each position.
        q_values = self.fc(lstm_out)    # shape: (batch_size, seq_length, num_codons)
        
        # Flatten the output to represent the Q-value for every possible (position, codon) action.
        q_values = q_values.view(state.size(0), self.seq_length * self.num_codons)
        return q_values