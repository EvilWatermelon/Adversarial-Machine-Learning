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
						filename='measurement.log',
						level=logging.DEBUG)
	logging.getLogger(__name__)

	logging_levels[logging_levelname](msg_string)
