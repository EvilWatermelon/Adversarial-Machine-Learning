import tracemalloc
import psutil

from gpuinfo import GPUInfo
from metrics.log import *

def start_ram_monitoring():
    # This function should be called right after the imports.
    tracemalloc.start()

def ram_resources():
    """
    Getting the current and peak memory usage when the program has finished.
    The output is passed to the log function.
    """
    current, peak = tracemalloc.get_traced_memory()
    log(f"Current memory usage: {current / 10**6}MB; Peak memory usage: {peak / 10**6}MB", "INFO")
    tracemalloc.stop()

def cpu_resources():
    """
    Getting the current CPU usage. The status is read exactly where the function is called.
    The output is passed to the log function.
    Using psuil for Windows and psutill for Linux.
    """
    ml_process = psutil.Process()

    cpu = ml_process.cpu_percent(interval=1.0)

    log(f"Current CPU usage: {cpu}%")

def gpu_resources():
    """
    Getting the current GPU usage. The status is read exactly where the function is called.
    The output is passed to the log function.
    This function works with Linux and Windows. Mac is not tested.
    """

    available_device = GPUInfo.check_empty()
    percent, memory = GPUInfo.gpu_usage()
    min_percent = percent.index(min([percent[i] for i in available_device]))
    min_memory = memory.index(min([memory[i] for i in available_device]))

    log(f"Percent: {min_percent}%, GPU memory: {min_memory}")
