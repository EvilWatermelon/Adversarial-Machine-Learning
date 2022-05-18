import tracemalloc
import psutil

from gpuinfo import GPUInfo
from metrics.log import *

class Attacker:
    # Computational resources
    def start_ram_monitoring(self):
        # This function should be called right after the imports.
        tracemalloc.start()

    def ram_resources(self):
        """
        Getting the current and peak memory usage when the program has finished.
        The output is passed to the log function.
        """
        current, peak = tracemalloc.get_traced_memory()
        log(f"Current memory usage: {current / 10**6}MB; Peak memory usage: {peak / 10**6}MB", "INFO")
        tracemalloc.stop()

    def cpu_resources(self):
        """
        Getting the current CPU usage. The status is read exactly where the function is called.
        The output is passed to the log function.
        Using psutil for Windows and psutill for Linux.
        """
        ml_process = psutil.Process()

        cpu = ml_process.cpu_percent(interval=1.0)

        log(f"Current CPU usage: {cpu}%")

    def gpu_resources(self):
        """
        Getting the current GPU usage of all GPUs. The status is read exactly where the function is called.
        The output is passed to the log function.
        This function works with Linux and Windows. Mac is not tested.
        """

        available_device = GPUInfo.check_empty()
        percent, memory = GPUInfo.gpu_usage()

        min_percent = percent.index(min([percent[i] for i in available_device]))
        min_memory = memory.index(min([memory[i] for i in available_device]))

        log(f"Percent: {min_percent}%, GPU memory: {min_memory}")

    def attackers_knowledge(self, attack):
        log("")

    def attackers_goal(self):
        log("")

class Attack:
    def positive_negative_label(self):
        log("")

    def attack_time(self):
        log("")

    def accuracy_log(self, true_values, predictions, normalize=False):

	    accuracy = np.sum(np.equal(true_values, predictions)) / len(true_values)

	    if normalize:
		    log(f"Accurary: {accuracy}")
	    else:
		    log(f"Normalized accurary: {np.mean(accuracy)}")

    def attack_specificty(self):
        log("")

    def training_data(x):
        log("")
