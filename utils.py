import time
import pynvml
import logging
import torch
import psutil
import os

def get_gpu_processes(pname='python'):
    """
    Checks for Python processes currently using an NVIDIA GPU, excluding the current process.

    Returns:
        A list of dictionaries, where each dictionary contains information
        about a Python process found on a GPU. Returns an empty list if none are found.
        Example:
        [
            {
                'pid': 1768748,
                'name': 'python',
                'gpu_index': 0,
                'used_gpu_memory_mb': 22286
            }
        ]
    """
    python_processes_on_gpu = []
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # We check for compute processes, as this is the typical use case for ML/DL.
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        
        for p in procs:
            try:
                # We need psutil to get the process name from its PID.
                proc_info = psutil.Process(p.pid)
                
                # IMPORTANT: Exclude the current script from the check.
                if p.pid == os.getpid():
                    continue
                
                # Check if the process name is 'python' or 'python3'.
                if pname in proc_info.name().lower():
                    python_processes_on_gpu.append({
                        'pid': p.pid,
                        'name': proc_info.name(),
                        'used_gpu_memory_GB': p.usedGpuMemory // (1024**3) # Bytes to MiB
                    })

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have ended, or we don't have permission to inspect it.
                continue

    except pynvml.NVMLError as e:
        logging.error(f"NVML Error: {e}. Is an NVIDIA driver installed and running?")
    finally:
        # Ensure NVML is shut down properly.
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    return python_processes_on_gpu

def wait_for_python_gpu_processes(poll_interval_minutes: float = 3, logger=None):
    """
    Waits until no other Python processes are using any NVIDIA GPU.

    This function periodically checks for Python processes running on any GPU
    and waits until all of them have completed. It ignores the current
    process running this script.

    Args:
        poll_interval_minutes (int): The interval in minutes to wait between checks.
    """
    log = logger or logging
    while True:
        # Get a list of other python processes on the GPU.
        other_py_procs = get_gpu_processes(pname='python')

        if not other_py_procs:
            log.info("-> No other Python processes found on the GPU. It is safe to proceed.")
            break
        else:
            # Construct a clear message showing what we're waiting for.
            # pids = [p['pid'] for p in other_py_procs]
            log.warning(
                "-> Waiting for other Python GPU process(es) to finish. "
                f"Found PIDs: {other_py_procs}. "
                f"Checking again in {poll_interval_minutes} minutes."
            )
            time.sleep(poll_interval_minutes*60)

def wait_for_gpu_memory(target_usage_gb: int = 20, poll_interval_minutes: float = 1, logger=None):
    """Waits until the GPU memory usage on device 0 is below a specified threshold.

    This function is designed for a single-host, single-GPU setup.

    Args:
        target_usage_gb (int): The target GPU memory usage in Gigabytes (GB). 
                               The script will wait until usage is below this value.
        poll_interval_minutes (int): The interval in minutes to wait between checks.
    """
    log = logger or logging
    if not torch.cuda.is_available():
        log.warning("CUDA not available. Skipping GPU memory check.")
        return

    try:
        pynvml.nvmlInit()
        # Assuming a single GPU, so we get a handle to device 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        threshold_bytes = target_usage_gb * (1024 ** 3)

        while True:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / (1024 ** 3)

            if mem_info.used < threshold_bytes:
                log.info(
                    f"GPU memory usage is {used_gb:.2f} GB, which is below the "
                    f"{target_usage_gb} GB threshold. Proceeding with training."
                )
                break
            else:
                log.warning(
                    f"Waiting for GPU memory to become available. "
                    f"Current usage: {used_gb:.2f} GB. "
                    f"Target: < {target_usage_gb} GB. "
                    f"Checking again in {poll_interval_minutes} minutes."
                )
                time.sleep(poll_interval_minutes * 60)
    except pynvml.NVMLError as e:
        log.error(f"An NVML error occurred: {e}. Could not monitor GPU memory. "
                      "Proceeding without memory check.")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass  # Can be ignored if already shut down or failed to init