import logging

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
