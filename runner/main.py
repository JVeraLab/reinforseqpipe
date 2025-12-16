# runner/main.py

#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import math
import os
import json
from datetime import datetime
import time
import argparse

# Importing modules from other files
from core.encoding import CodonEmbedding, Verification
from core.env import MRNAEnv, compute_allowed_mask
from core.agent import DQNAgent
from data.assemble import Assemble
from data.read_fasta import read_fasta_sequences

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
        max_steps=max_steps, 
        factor=factor,
        mask_last_n_codons=mask_last_n_codons 
    )
    
    initial_cai, initial_mfe = env.get_initial_state()
    seq_length = len(original_mrna)
    
    # Hyperparameters (preserved from original script)
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
    # episode_number: list[int] = []
    # episode_cai: list[float] = []
    # episode_mfe: list[float] = []
    # episode_score: list[float] = []

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
        
        # episode_number.append(episode)
        # episode_cai.append(current_cai)
        # episode_mfe.append(current_mfe)
        # episode_score.append(current_score)
        
        # 8. Update best sequence (Minimize the score)
        if current_score < best_score:
            best_cai = current_cai
            best_mfe = current_mfe
            best_score = current_score
            best_sequence = current_sequence
            best_sequences.append(best_sequence)
        
    # The original code structure returns the calculated GC content from env, so it needs to be accessible.
    env_with_gc = MRNAEnv(original_mrna, weights, computing_codon.compute_codon_vector, 
                          max_steps, factor, mask_last_n_codons)

    return (initial_cai, initial_mfe, env_with_gc.calculate_gc_content(original_dna), # Original metrics
            best_sequence, best_cai, best_mfe, env_with_gc.calculate_gc_content(best_sequence))

# ==========================================================================================================================================================
# Worker and Multiprocessing
# ==========================================================================================================================================================
def worker(sequence: str, device: torch.device, weights: dict, num_episodes: int, 
           max_steps: int, index: int, factor: float, mask_last_n_codons: int) -> tuple[int, dict] | None:
    """
    Worker function executed in a separate process to run the DQN training loop 
    for a single sequence and gather the results.
    """
    print(f"Worker {index} starting.")
    start_time = time.perf_counter()
    
    # Run the training loop
    (original_dna_cai, original_dna_mfe, original_dna_gc_content,
     best_sequence, best_sequence_cai, best_sequence_mfe, best_sequence_gc_content) = train_agent_dqn(
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
        "Original CAI": round(original_dna_cai, 4), 
        "Original MFE": round(original_dna_mfe, 4),
        "Original GC content": round(original_dna_gc_content, 4),
        "Final CAI": round(best_sequence_cai, 4), 
        "Final MFE": round(best_sequence_mfe, 4), 
        "Final GC content": round(best_sequence_gc_content, 4),
        "Original MFECAIλ score": round(orig_mfe_cai_score, 4),
        "Final MFECAIλ score": round(final_mfe_cai_score, 4),
        "The gain of CAI score:": round(best_sequence_cai - original_dna_cai, 4),
        "The gain of MFE score:": round(original_dna_mfe - best_sequence_mfe, 4),
        "The gain of MFECAIλ score:": round(original_dna_mfe - best_sequence_mfe, 4),
        "Peptide_compatibility": bool(verified),
        "Time Taken (s)": round(time_taken, 4)
    }
    return (index, results_row)


def main(device, weights_path: str, factor: float, episodes: int, batch_size: int, 
         sequences_path: str = 'sequences.json', generation: bool = False,
         fasta_filepath: str = None, ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set start method for multiprocessing compatibility with CUDA (required for multi-processing path)
    if batch_size > 1:
        mp.set_start_method('spawn', force=True) 

    # ... (rest of the setup prints and folder creation are kept as is) ...
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
    # Setup output folder
    folder_name = f'results'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 1. Sequence Generation/Loading Adaptation
    sequences: list[str] = []  # List of final DNA sequences for processing
    mask_last_n_codons: int = 0
    
    if fasta_filepath:
        print(f"Loading and processing sequences from FASTA file: {fasta_filepath}...")
        
        # Load data in format: {name: (sequence, type)}
        loaded_data = read_fasta_sequences(fasta_filepath)
        
        # Initialize Assemble instance for backtranslation
        assembler = None
        
        for seq_name, (seq, mol_type) in loaded_data.items():
            if mol_type == 'PEPTIDE':
                # Pre-processing 1: Backtranslate Peptide
                print(f"-> Backtranslating peptide: {seq_name}")
                if assembler is None:
                    # Initialize only if needed
                    assembler = Assemble(weights_path) 
                
                # NOTE: Need to ensure your backtranslation function returns the DNA sequence
                # For simplicity, let's assume 'backtranslate_single_peptide' is the function
                dna_sequence = assembler.backtranslate_single_peptide(seq) 
                sequences.append(dna_sequence)
                
            elif mol_type == 'RNA':
                # Pre-processing 2: RNA to DNA
                print(f"-> Converting RNA to DNA: {seq_name}")
                # NOTE: Need to ensure your rna_to_dna function is defined
                dna_sequence = seq.replace('U', 'T') # Simple conversion
                sequences.append(dna_sequence)
                
            else: # mol_type == 'DNA' or other
                # No processing needed
                sequences.append(seq)
                
    elif generation:
        print("Assembling sequences (generation=True)...")
        assembling = Assemble(weights_path) 
        sequences, _ = assembling.backtranslate_peptide() 
    else: 
        print(f"Loading sequences from JSON file: {sequences_path}...")
        with open(sequences_path, 'r') as file:
            sequences = json.load(file)

    # 2. Load CAI weights (Unmodified)
    with open(weights_path, 'r') as file:
        weights: dict = json.load(file)

    # ... (Rest of the function - Sections 2, 3, 4 - remains the same as the prior adapted version) ...
    
    if not isinstance(weights, dict):
        raise ValueError("weights should be a dictionary")
        
    mask_last_n_codons = 0 
    
    # 3. Processing (Parallel/Single-Process - logic remains the same)
    df_results = pd.DataFrame()
    num_episodes = episodes 
    
    total_sequences = len(sequences)
    print(f"Total sequences to process: {total_sequences}")
    
    i = 1 # Batch counter
    
    # Iterate through sequences in batches
    for batch_start in range(0, total_sequences, batch_size):
        # ... (Processing loop logic: if batch_size=1 use direct call, else use ProcessPoolExecutor) ...
        # (This section is identical to Section 3 in the previous response)
        
        batch_end = min(batch_start + batch_size, total_sequences)
        batch = sequences[batch_start:batch_end]
        
        print(f"Processing batch number {i} (sequences {batch_start} to {batch_end-1})")
        
        if batch_size == 1:
            # === Single-Processing Path (Batch size 1) ===
            sequence = batch[0]
            result = worker(sequence, device, weights, num_episodes, 
                            len(sequence), batch_start, factor, mask_last_n_codons)

            if result is not None and result[1] is not None:
                df_results = df_results._append(result[1], ignore_index=True) 
            else:
                print(f"Received None result for sequence in batch {i}") 

        else:
            # === Multi-Processing Path (Batch size > 1) ===
            # Ensure 'spawn' is set for multiprocessing compatibility with CUDA
            mp.set_start_method('spawn', force=True) 
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(worker, sequence, device, weights, num_episodes, 
                                    len(sequence), batch_start + j, factor, mask_last_n_codons)
                    for j, sequence in enumerate(batch)
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None and result[1] is not None:
                        df_results = df_results._append(result[1], ignore_index=True) 
                    else:
                        print(f"Received None result for one of the workers in batch {i}") 
            
            torch.cuda.empty_cache()
            time.sleep(2) 

        print(f"Finished batch number {i}")
        i += 1
    # 4. Save Results (Unmodified)
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{folder_name}/results_for_run_dated_{formatted_time}_{factor}.csv"
    df_results.to_csv(filename, index=True)
    print(f"Results saved to {filename}")

def parse_args():
    """Defines command-line arguments and returns the parsed Namespace object."""
    
    parser = argparse.ArgumentParser(
        description="DQN-based Codon Optimization Pipeline for mRNA vaccine optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Weights/Input File Path
    parser.add_argument(
        '-w', '--weights-path',
        type=str,
        default='../codon_usage_dict_linear_design.json',
        help='Path to the JSON file containing codon usage weights for CAI calculation.'
    )
    
    # 2. Episodes per Run 
    parser.add_argument(
        '-e', '--episodes',
        type=int,
        default=20,
        help='Number of optimization episodes to run for each input sequence.'
    )
    
    # 3. Lambda Factor (The default value is 4.0 as a balance between CAI and MFE)
    parser.add_argument(
        '-l', '--lambda',
        type=float,
        default=4.0, 
        help='Lambda factor in the objective function: |MFE - L * factor * log10(CAI)|.'
    )
    
    # 4. Batch Size 
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=1,
        help='Number of sequences to process in parallel (max workers).'
    )
    
    # 5. THis parameter states if the code uses its own 
    parser.add_argument(
        '-g', '--generation',
        action='store_false',
        dest='generation', # Stores the result in args.generation
        help='If True, pre-determined sequences are generated using the Assemble class. If False, input your sequence fasta file.'
    )
    
    # 6. It is possible to provide the optional fasta file with sequences
    parser.add_argument(
        '-i', '--fasta-input-path',
        type=str,
        default='../test.fasta',
        help='Optional path to a FASTA file containing input sequences. Overrides sequence generation.'
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    main(weights_path   = args.weightspath, 
         factor         = args.lambda_parameter, 
         episodes       = args.episodes, 
         batch_size     = args.batch_size, 
         generation     = args.generation,
         fasta_filepath = args.fasta_input_path)