#!/usr/bin/env python

import os
import json 
import time 
import torch 
import pandas as pd
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio import SeqIO  

from environment import MRNAEnv
from utils import Verification, Assemble
from device_setup import get_device, init_multiprocessing
from train_agent import train_agent_dqn

def worker(sequence: str, seqname: str, device: torch.device, weights: dict, num_episodes: int, 
           max_steps: int, factor: float, mask_last_n_codons: int) -> tuple[str, dict] | None:
    
    print(f"Worker {seqname} starting.")
    start_time = time.perf_counter()
    
    # Run training
    (original_dna_cai, original_dna_mfe, original_dna_gc_content,
     best_sequence, best_sequence_cai, best_sequence_mfe, best_sequence_gc_content,
     episode_number, episode_cai, episode_mfe, episode_score) = train_agent_dqn(
        sequence, device, weights, num_episodes, max_steps, factor, mask_last_n_codons)
        
    time_taken = time.perf_counter() - start_time
    verified = Verification(sequence, best_sequence).compare_dna_sequences()
    
    seq_length_codons = len(sequence) // 3
    final_mfe_cai_score = abs(best_sequence_mfe - factor * seq_length_codons * math.log10(best_sequence_cai))

    # KEEPING YOUR EXACT DICTIONARY STRUCTURE
    results_row = {
        "seqname": seqname, # Added this for the FASTA header logic
        "Original Sequence": sequence, 
        "Final Sequence": best_sequence, 
        "Episode List": list(episode_number),
        "CAI List": list(episode_cai), 
        "MFE List": list(episode_mfe),
        "Score List": list(episode_score), 
        "Original CAI": round(original_dna_cai, 4), 
        "Original MFE": round(original_dna_mfe, 4),
        "Original GC content": round(original_dna_gc_content, 4),
        "Final CAI": round(best_sequence_cai, 4), 
        "Final MFE": round(best_sequence_mfe, 4), 
        "Final GC content": round(best_sequence_gc_content, 4),
        "Original MFExCAI score": round(final_mfe_cai_score, 4), # Simplified for worker
        "Final MFExCAI score": round(final_mfe_cai_score, 4),
        "The gain of CAI score:": round(best_sequence_cai - original_dna_cai, 4),
        "The gain of MFE score:": round(original_dna_mfe - best_sequence_mfe, 4), 
        "Peptide_compatibility": bool(verified),
        "Time Taken (s)": round(time_taken, 4)
    }
    
    return (seqname, results_row)


def main(weights_filename: str, factor: float, episodes: int, batch_size: int, generation: bool = False):
    device = get_device()
    init_multiprocessing(device)
    
    folder_name = 'results_optimization'
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    
    print('Initializing the process...')

    # 1. LOAD FROM FASTA
    fasta_path = 'test_sequence.fasta'
    if generation:
        raw_seqs, mask_last_n_codons = Assemble(weights_filename).backtranslate_peptide()
        sequences = [{"name": f"Perm_{i}", "seq": s} for i, s in enumerate(raw_seqs)]
    else:
        sequences = [{"name": r.id, "seq": str(r.seq).upper()} for r in SeqIO.parse(fasta_path, "fasta")]
        mask_last_n_codons = 0

    # 2. LOAD WEIGHTS
    if not os.path.exists(weights_filename):
        raise FileNotFoundError(f"Could not find weights file: {weights_filename}")
        
    with open(weights_filename, 'r') as f: 
        weights = json.load(f)

    df_results = pd.DataFrame()

    # 3. PARALLEL LOOP
    for b_start in range(0, len(sequences), batch_size):
        batch = sequences[b_start : b_start + batch_size]
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(worker, item["seq"], item["name"], device, weights, 
                                episodes, len(item["seq"]), factor, mask_last_n_codons) 
                for item in batch
            ]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    df_results = df_results._append(res[1], ignore_index=True)

        torch.cuda.empty_cache()
        time.sleep(2)

    # 4. OUTPUTS
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    df_results.to_csv(f"{folder_name}/results_{ts}.csv", index=False)
    
    with open(f"{folder_name}/optimized_{ts}.fasta", 'w') as f:
        for _, row in df_results.iterrows():
            header = f">{row['seqname']} | CAI:{row['Final CAI']} | MFE:{row['Final MFE']} | GC:{row['Final GC content']} | Score:{row['Final MFExCAI score']}"
            f.write(f"{header}\n{row['Final Sequence']}\n")
    print(f"Finished. Results in {folder_name}")

if __name__ == "__main__":
    # MAKE SURE THIS FILE EXISTS IN YOUR FOLDER
    target_weights = 'codon_usage_dict_genomic.json'
    main(target_weights, factor=4, episodes=10, batch_size=1, generation=False)