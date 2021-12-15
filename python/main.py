from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

import logging

cancer = datasets.load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"Precision: {metrics.precision_score(y_test, y_pred)}")
print(f"Recall: {metrics.recall_score(y_test, y_pred)}")

def log(message, logging_levelname='INFO'):
	msg_string = str(f'{message}')

	logging.basicConfig(format='[%(levelname)s] [%(asctime)s]: %(message)s',
						datefmt='%m/%d/%Y %I:%M:%S %p',
						filename='example.log',
						encoding='utf-8',
						level=logging.DEBUG)

	levelname_list = {'DEBUG':logging.debug(msg_string),
					  'INFO':logging.info(msg_string), 
					  'WARNING':logging.warning(msg_string), 
					  'ERROR':logging.error(msg_string), 
					  'CRITICAL':logging.critical(msg_string)}

	#TODO:call only logging_levelname function
	if levelname_list.get(logging_levelname) != None:
		levelname_list[logging_levelname]

log(metrics.recall_score(y_test, y_pred), 'DEBUG')
