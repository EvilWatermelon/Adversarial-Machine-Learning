import tracemalloc
import psutil

from gpuinfo import GPUInfo
from metrics.log import *

counter = 0

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
# threats to validity
    def clean_label(self):

        log("Clean Label Backdoor Attack")

        log("")

    def hidden_trigger(self):

        log("Hidden Trigger Backdoor Attack")


        log("")

    def pattern_backdoor(self):
        log("Pattern Backdoor Attack")

    def attackers_knowledge(self, attack):

        log("Measuring attacker's knowledge...")

        attacks = {
            "clean_label": self.clean_label,
            "hidden_trigger": self.hidden_trigger,
            "pattern_backdoor": self.pattern_backdoor
        }

        attacks[attack]()

    def attackers_goal(self):
        log("")

class Attack:
    def positive_negative_label(self, thresholds=None, name=None, dtype=None):
        tp = tf.keras.metrics.TruePositives(thresholds, name, dtype)
        tn = tf.keras.metrics.TrueNegatives(thresholds, name, dtype)
        fp = tf.keras.metrics.FalsePositives(thresholds, name, dtype)
        fn = tf.keras.metrics.FalseNegatives(thresholds, name, dtype)

        return tp, tn, fp, fn


    def attack_time(self, attack):
        """
        0: training times
        1: test time
        """

        attacks = {
            "clean_label": 0,
            "hidden_trigger": 0,
            "pattern_backdoor": 0
        }

        log("")

    def accuracy_log(self, true_values, predictions, normalize=False):

	    accuracy = np.sum(np.equal(true_values, predictions)) / len(true_values)
        normalization = np.mean(accuracy)

	    if normalize:
		    log(f"Accurary: {accuracy}")
            return accuracy
	    else:
		    log(f"Normalized accurary: {normalization}")
            return normalization

    def attack_specificty(self, target=False):
        if target:
            log("Untargeted attack")
            return 0
        else:
            log("Targeted attack")
            return 1
