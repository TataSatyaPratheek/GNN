#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gc
import os
import psutil
import time
from colorama import Fore, Style, init

# Initialize colorama
init()

def clear_gpu_memory():

    print(f"{Fore.CYAN}Clearing GPU memory...{Style.RESET_ALL}")
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        # Get initial memory stats
        initial_allocated = torch.cuda.memory_allocated() / (1024**2)
        initial_reserved = torch.cuda.memory_reserved() / (1024**2)
        
        print(f"Initial GPU memory: {initial_allocated:.1f} MB allocated, {initial_reserved:.1f} MB reserved")
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Synchronize device
        torch.cuda.synchronize()
        
        # Get final memory stats
        final_allocated = torch.cuda.memory_allocated() / (1024**2)
        final_reserved = torch.cuda.memory_reserved() / (1024**2)
        
        print(f"Final GPU memory: {final_allocated:.1f} MB allocated, {final_reserved:.1f} MB reserved")
        print(f"Freed {initial_reserved - final_reserved:.1f} MB")
    else:
        print(f"{Fore.YELLOW}No GPU available{Style.RESET_ALL}")
    
    # Also report CPU memory
    cpu_memory = psutil.virtual_memory()
    print(f"CPU memory: {cpu_memory.percent}% used, {cpu_memory.available / (1024**3):.2f} GB available")

if __name__ == "__main__":
    clear_gpu_memory()
    
    # Also kill any orphaned CUDA processes if on Linux
    if os.name == 'posix':
        try:
            os.system("nvidia-smi | grep 'python' | awk '{print $3}' | xargs -r kill -9")
            print(f"{Fore.GREEN}Killed orphaned CUDA processes{Style.RESET_ALL}")
        except:
            pass
