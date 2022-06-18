import logging
import numpy as np
from measurement import monitoring

def log(message):
	msg_string = str(f'{message}')

	logging.basicConfig(format='%(levelname)s - %(name)s - %(asctime)s: %(message)s',
						datefmt='%m/%d/%Y %I:%M:%S %p',
						filename='measurement.log',
						level=logging.getLogger('monitoring').setLevel(logging.INFO))
