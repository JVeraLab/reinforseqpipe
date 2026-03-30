import torch
import multiprocessing as mp

def get_device():
    # Detect Nvidia (CUDA)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Detect Apple Silicon (MPS)
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    # Default to CPU
    return torch.device("cpu")

def init_multiprocessing(device):
    # 'spawn' is required for CUDA/MPS compatibility in parallel loops
    if device.type in ["cuda", "mps"]:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass