#!/usr/bin/env python
# https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
import subprocess, re
import os

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Could not parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def get_free_gpu_memory():
    """Returns a list of free GPU memory in MiB, ordered by GPU ID."""
    output = run_command("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    free_memory = [int(line.strip()) for line in output.strip().split("\n")]
    return free_memory

def gpu_memory_map():
    """Returns map of GPU id to free memory on that GPU."""
    free_memory = get_free_gpu_memory()
    gpus = list_available_gpus()
    return {gpu_id: free_memory[i] for i, gpu_id in enumerate(gpus)}

def pick_gpu_lowest_memory(gpus_avail=None):
    """
    Returns the GPU with the most available memory (least used).
    Considers CUDA_VISIBLE_DEVICES if defined, and applies specific mapping for 'simons-2'.
    """
    import os
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpus_avail = [int(x) for x in cuda_visible_devices.split(",")]
        # print(f"CUDA_VISIBLE_DEVICES is set. Considering GPUs: {gpus_avail}")
    

    # GPU ID mapping for 'simons-2'
    simons2_nv_to_torch_map = {0: 0, 5: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    if "simons-2" in os.uname().nodename:
        nv_to_torch_map = simons2_nv_to_torch_map
    else:
        nv_to_torch_map = {i: i for i in range(16)}  # Identity mapping for all possible GPUs

    # Filter mapping based on available GPUs
    if gpus_avail is not None:
        nv_to_torch_map = {k: v for k, v in nv_to_torch_map.items() if k in gpus_avail}

    best_gpu_torch = None
    best_memory = None
    try:
        # Get free memory map (nvidia-smi output)
        free_memory_map = gpu_memory_map()
        print(f"Free memory map: {free_memory_map}")

        # Filter GPUs in the memory map based on mapping
        mem_id_list = [(free_memory_map[gpu_id], gpu_id) for gpu_id in nv_to_torch_map.keys() if gpu_id in free_memory_map]
        print(f"Filtered memory-ID list: {mem_id_list}")

        # Sort GPUs by memory in descending order
        if mem_id_list:
            best_memory, best_gpu_nv = max(mem_id_list, key=lambda x: x[0])
            best_gpu_torch = nv_to_torch_map[best_gpu_nv]
            print(f"Best GPU (nvidia-smi ID): {best_gpu_nv}, Best GPU (torch ID): {best_gpu_torch}, Free Memory: {best_memory} MiB")
        else:
            print("No available GPUs meet the memory requirements.")
    except Exception as e:
        print(f"Error while selecting GPU: {e}")

    return best_gpu_torch, best_memory
if __name__ == "__main__":

    # argument is list of GPU IDs to consider
    import sys
    if len(sys.argv) > 1:
        gpu_avail = [int(x) for x in sys.argv[1:]]
    else:
        gpu_avail = None

    best_gpu, avail_mem = pick_gpu_lowest_memory(gpu_avail)
    print(f"Best GPU: {best_gpu}, Free Memory: {avail_mem} MiB")
    