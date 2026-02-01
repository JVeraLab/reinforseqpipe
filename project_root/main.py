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
import argparse
from utils import Verification, Assemble
from device_setup import get_device, init_multiprocessing
from train_agent import train_agent_dqn

#!/usr/bin/env python

import os
import json 
import time 
import torch 
import argparse 
import pandas as pd
import math
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from Bio import SeqIO  
import psutil
from environment import MRNAEnv
from utils import Verification, Assemble
from device_setup import get_device, init_multiprocessing
from train_agent import train_agent_dqn

def parse_args():
    """Handles CLI inputs with shortened flags."""
    parser = argparse.ArgumentParser(description="ReinforSeqPipe CLI")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to weights JSON")
    parser.add_argument("-f", "--fasta", type=str, required=True, help="Path to input FASTA")
    parser.add_argument("-l", "--lam", type=float, default=4.0, help="Lambda factor (default: 4.0)")
    parser.add_argument("-e", "--episodes", type=int, default=20, help="Episodes (default: 20)")
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Parallel processes (default: 1)")
    parser.add_argument("-g", "--gen", action="store_true", default = False, help="Enable generation mode")
    return parser.parse_args()


def get_smart_parallel_count(requested_p):
    # 1. Physical CPU Core Limit
    cpu_count = os.cpu_count() or 1
    
    # 2. RAM Limit (assuming ~800MB per worker)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    ram_limit = int(available_ram_gb / 0.8) # 0.8GB buffer per worker
    
    # 3. GPU VRAM Limit (if using CUDA)
    vram_limit = requested_p
    if torch.cuda.is_available():
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Assuming each LSTM worker takes ~1.5GB of VRAM
        vram_limit = int(gpu_mem_total / 1.5)

    smart_limit = min(cpu_count, ram_limit, vram_limit, requested_p)
    
    return max(1, smart_limit)

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
        "seqname": seqname, 
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
        "Original MFExCAI score": round(final_mfe_cai_score, 4), 
        "Final MFExCAI score": round(final_mfe_cai_score, 4),
        "The gain of CAI score:": round(best_sequence_cai - original_dna_cai, 4),
        "The gain of MFE score:": round(original_dna_mfe - best_sequence_mfe, 4), 
        "Peptide_compatibility": bool(verified),
        "Time Taken (s)": round(time_taken, 4)
    }
    
    return (seqname, results_row)

def main():
    args = parse_args()
    device = get_device()
    init_multiprocessing(device)
    


    folder_name = 'results_optimization'
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    
    if args.gen:
        raw_seqs, mask_last_n_codons = Assemble(args.weights).backtranslate_peptide()
        sequences = [{"name": f"Perm_{i}", "seq": s} for i, s in enumerate(raw_seqs)]
    else:
        sequences = [{"name": r.id, "seq": str(r.seq).upper()} for r in SeqIO.parse(args.fasta, "fasta")]
        mask_last_n_codons = 0
    
    
    original_p = args.parallel
    args.parallel = get_smart_parallel_count(original_p)

    if args.parallel < original_p:
        print(f"!!! MEMORY/CPU LIMIT: Scaling down from {original_p} to {args.parallel} workers.")

    with open(args.weights, 'r') as f: 
        weights_dict = json.load(f)

    df_results = pd.DataFrame()

    # Parallel batching loop
    for b_start in range(0, len(sequences), args.parallel):
        batch = sequences[b_start : b_start + args.parallel]
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(worker, item["seq"], item["name"], device, weights_dict, 
                                args.episodes, len(item["seq"]), args.lam, mask_last_n_codons) 
                for item in batch
            ]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    df_results = df_results._append(res[1], ignore_index=True)

    # Output files
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    df_results.to_csv(f"{folder_name}/results_{ts}.csv", index=False)
    
    with open(f"{folder_name}/optimized_{ts}.fasta", 'w') as f:
        for _, row in df_results.iterrows():
            header = f">{row['seqname']} | CAI:{row['Final CAI']} | MFE:{row['Final MFE']} | Score:{row['Final MFExCAI score']}"
            f.write(f"{header}\n{row['Final Sequence']}\n")
    print(f"Done. Files saved in {folder_name}")

if __name__ == "__main__":
    main()