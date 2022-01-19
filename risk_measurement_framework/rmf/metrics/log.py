import logging
import numpy as np

def log(message, logging_levelname: str = 'INFO'):
	msg_string = str(f'{message}')

	logging_levels = {
		'INFO': logging.info,
		'DEBUG': logging.debug,
		'WARNING': logging.warning,
		'ERROR': logging.error,
		'CRITICAL': logging.critical
	}

	logging.basicConfig(format='%(levelname)s - %(name)s - %(asctime)s: %(message)s',
						datefmt='%m/%d/%Y %I:%M:%S %p',
						filename='example.txt',
						level=logging.DEBUG)

	logging_levels[logging_levelname](msg_string)

def accuracy_log(true_values, predictions, normalize=False):

	accuracy = np.sum(np.equal(true_values, predictions)) / len(true_values)

	if normalize:
		log(f"Accurary: {accuracy}")
	else:
		log(f"Normalized accurary: {np.mean(accuracy)}")

def precision_log(true_values, predictions):
	TP = ((predictions == 1) & (true_values == 1)).sum()
	FP = ((predictions == 1) & (true_values == 0)).sum()

	precision = TP / (TP + FP)

	log(f"Precision of the model: {precision}")
