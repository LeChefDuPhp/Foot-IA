import psutil
import torch
import os

def get_optimal_config():
    """
    Detects system hardware and returns optimal hyperparameters.
    """
    config_overrides = {}
    
    # 1. CPU Cores -> PARALLEL_ENVS
    # Use logical cores but leave some for system/rendering
    cpu_count = psutil.cpu_count(logical=True)
    if cpu_count:
        # Use 75% of cores, min 1, max 64
        optimal_envs = max(1, int(cpu_count * 0.75))
        config_overrides['PARALLEL_ENVS'] = optimal_envs
    else:
        cpu_count = "Unknown"

    # 2. RAM -> MAX_MEMORY
    # Estimate: 1M steps ~ 1-2GB RAM depending on state size
    # Let's be safe: 1GB per 100k steps? No, that's too high.
    # Replay buffer is deque of tuples.
    # Let's assume 16GB RAM -> 1M steps is safe.
    total_ram = psutil.virtual_memory().total # in bytes
    total_ram_gb = total_ram / (1024**3)
    
    # Scale: 1M steps per 16GB, min 100k
    optimal_memory = int((total_ram_gb / 16.0) * 1_000_000)
    optimal_memory = max(100_000, optimal_memory)
    config_overrides['MAX_MEMORY'] = optimal_memory

    # 3. GPU VRAM -> BATCH_SIZE
    # If CUDA, scale batch size. If CPU, keep small.
    device_name = "CPU"
    if torch.cuda.is_available():
        try:
            vram = torch.cuda.get_device_properties(0).total_memory # bytes
            vram_gb = vram / (1024**3)
            device_name = torch.cuda.get_device_name(0)
            
            # Scale: 4GB -> 2048, 8GB -> 4096, 16GB -> 8192
            # Base: 512 per GB?
            # 4GB * 512 = 2048. Correct.
            optimal_batch = int(vram_gb * 512)
            
            # Round to nearest power of 2 for efficiency (optional but good)
            # Just keep it simple for now, maybe cap it.
            optimal_batch = max(1024, optimal_batch)
            config_overrides['BATCH_SIZE'] = optimal_batch
        except:
            config_overrides['BATCH_SIZE'] = 1024 # Fallback
    else:
        # CPU Training
        config_overrides['BATCH_SIZE'] = 256 # Smaller for CPU
        
    print(f"--- Auto-Tuning Hardware ---")
    print(f"CPU Cores: {cpu_count}")
    print(f"RAM: {total_ram_gb:.2f} GB")
    print(f"Device: {device_name}")
    print(f"----------------------------")
    
    return config_overrides
