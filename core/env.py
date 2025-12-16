# core/env.py

import torch
import numpy as np
import math
from CAI import CAI
import RNA # ViennaRNA
from .encoding import CodonEmbedding

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

        # Standard genetic code for internal use
        self.codon_embedder = CodonEmbedding()
        self.genetics_code = self.codon_embedder.genetic_code
        self.codons = list(self.genetics_code.keys())
        self.synonymous_dict = self.codon_embedder._compute_synonymous_groups()
        
        self.prev_cai: float = 0.0 # will be set during reset
        self.prev_mfe: float = 0.0 # will be set during reset

        self.embedding_function = embedding_function
        self.weights = weights

    def calculate_cai(self, mrna_sequence: str) -> float:
        """Calculates CAI (CAI expects DNA string, T's)."""
        return CAI(mrna_sequence, weights=self.weights)

    def calculate_mfe(self, mrna_sequence: str) -> float:
        """Calculates MFE (ViennaRNA expects RNA string, U's)."""
        # 
        return RNA.fold(mrna_sequence.replace("T", "U"))[1]

    def calculate_gc_content(self, mrna_sequence: str) -> float:
        """Calculates the GC content of the sequence."""
        gc_count = mrna_sequence.count('G') + mrna_sequence.count('C')
        return gc_count / len(mrna_sequence)

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
        info = {}

        # 1. Check for illegal (non-synonymous) move
        if old_aa != new_aa:
            # Illegal move: non-synonymous substitution attempted 
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
        old_score = self.calculate_mfecaiscore(self.prev_mfe, self.prev_cai, self.length_codons, self.factor)
        new_score = self.calculate_mfecaiscore(new_mfe, new_cai, self.length_codons, self.factor)
        reward = old_score - new_score # Reward is the decrease in score (since we minimize the score).
        
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
        allowed = [codon for codon in syn_codons if codon != current_codon]
        
        # If no synonymous alternatives exist (e.g., for ATG/Trp), allow the no-op (current codon)
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
        current_codon = env.current_mrna[i]
        
        # is_fixed is True if the position is Met/Trp OR it's in the C-terminal mask.
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