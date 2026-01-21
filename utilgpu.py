#!/usr/bin/env python
# https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
import subprocess, re
import os
import numpy 
import torch

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

_gpu_mapping_cache = None

def pick_gpu_lowest_memory(gpus_avail=None):
    """
    Returns the GPU with the most available memory (least used).
    Considers CUDA_VISIBLE_DEVICES if defined
    """
    global _gpu_mapping_cache
    if _gpu_mapping_cache is not None:
        nv_to_torch_map = _gpu_mapping_cache
    else:
        nv_to_torch_map = get_gpu_mapping()
        print(f"GPU mapping (nvidia-smi ID to torch ID): {nv_to_torch_map}")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpus_avail = [int(x) for x in cuda_visible_devices.split(",")]
        print(f"CUDA_VISIBLE_DEVICES is set. Considering GPUs: {gpus_avail}")
    


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
        mem_nid_list = [(free_memory_map[gpu_id], gpu_id, nv_to_torch_map[gpu_id]) for gpu_id in nv_to_torch_map.keys() if gpu_id in free_memory_map]
        # sort by memory in descending order
        mem_nid_list.sort(reverse=True, key=lambda x: x[0])
        print(f"Filtered memory-nid-tid list: {mem_nid_list}")

        

        # Sort GPUs by memory in descending order
        if mem_nid_list:
            best_memory, best_gpu_nv, best_gpu_torch = max(mem_nid_list, key=lambda x: x[0])
            print(f"Best GPU (nvidia-smi ID): {best_gpu_nv}, Best GPU (torch ID): {best_gpu_torch}, Free Memory: {best_memory} MiB")
        else:
            print("No available GPUs meet the memory requirements.")
    except Exception as e:
        print(f"Error while selecting GPU: {e}")

    return best_gpu_torch, best_memory

def get_gpu_mapping():
    """
    Returns a dictionary mapping from the nvidia-smi GPU ID to the corresponding torch device ID,
    using the UUID for matching.
    """
    try:
        # Get output from nvidia-smi -L
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
    except Exception as e:
        raise RuntimeError("Error executing nvidia-smi: " + str(e))

    # Parse the nvidia-smi output.
    # Example line:
    # GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-b842f6eb-4163-94db-35fe-192180b5feb8)
    pattern = re.compile(r"GPU\s+(\d+):.*\(UUID:\s+GPU-([\w\-]+)\)")
    smi_id_to_uuid = {}
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            smi_id = int(match.group(1))
            # Normalize UUID (nvidia-smi includes a "GPU-" prefix which we remove)
            uuid = match.group(2).lower()
            smi_id_to_uuid[smi_id] = uuid

    # Now iterate over torch devices and build the mapping.
    mapping = {}
    for torch_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(torch_id)
        torch_uuid = str(props.uuid).lower()
        # Find matching nvidia-smi GPU id
        for smi_id, uuid in smi_id_to_uuid.items():
            if uuid == torch_uuid:
                mapping[smi_id] = torch_id
                break

    return mapping


if __name__ == "__main__":

    # argument is list of GPU IDs to consider
    import sys
    if len(sys.argv) > 1:
        gpu_avail = [int(x) for x in sys.argv[1:]]
    else:
        gpu_avail = None

    
    best_gpu_torch, avail_mem = pick_gpu_lowest_memory(gpu_avail)
    print(f"Best GPU: {best_gpu_torch}, Free Memory: {avail_mem} MiB")
