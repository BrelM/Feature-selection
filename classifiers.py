'''
	algorithms.py

	Implementation of some ML classifiers.


	By Alph@B, AKA Brel MBE

'''


import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis







def svm_classifier(data:pd.DataFrame, y:pd.Series, columns:pd.Series | list=None)->pd.DataFrame:
	'''
		SVM one-vs-one and one-vs-rest classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
	   	- columns: vector-like object. The columns to use during classification. Defaults to all.
	   	## Returns
	   	- A matrix-like object consisting of performance metrics. 
	'''
	

	if y.cat.categories.shape[0] > 2:
		model = svm.SVC(kernel='linear', C=1)
	
	else:
		model = svm.SVC(kernel='linear', decision_function_shape='ovo', C=1)

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)


	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return "{:.2}, {:.2}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

	# scores = cross_validate(model, data.to_numpy(), y.to_numpy(), cv=5, scoring=["accuracy"])

	# return "{:.2}".format(np.mean(scores['test_accuracy']))
	# return "{:.2}, {:.2}".format(np.mean(scores['test_accuracy']), np.mean(scores['test_recall']))



def logreg_classifier(data:pd.DataFrame, y:pd.Series, columns:pd.Series | list=None)->pd.DataFrame:
	'''
		Logistic regression classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
	   	- columns: vector-like object. The columns to use during classification. Defaults to all.
	   	## Returns
	   	- A matrix-like object consisting of performance metrics. 
	'''
	
	model = LogisticRegression()

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)


	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return "{:.2}, {:.2}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))


	# scores = cross_validate(model, data.to_numpy(), y.to_numpy(), cv=5, scoring=["accuracy", "recall"])

	# return "{:.2}, {:.2}".format(np.mean(scores['test_accuracy']), np.mean(scores['test_recall']))



def dectree_classifier(data:pd.DataFrame, y:pd.Series, columns:pd.Series | list=None)->pd.DataFrame:
	'''
		Decision tree classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
	   	- columns: vector-like object. The columns to use during classification. Defaults to all.
	   	## Returns
	   	- A matrix-like object consisting of performance metrics. 
	'''
	
	model = tree.DecisionTreeClassifier()

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return "{:.2}, {:.2}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

	# scores = cross_validate(model, data.to_numpy(), y.to_numpy(), cv=5, scoring=["accuracy", "recall"])

	# return "{:.2}, {:.2}".format(np.mean(scores['test_accuracy']), np.mean(scores['test_recall']))



def randforest_classifier(data:pd.DataFrame, y:pd.Series, columns:pd.Series | list=None)->pd.DataFrame:
	'''
		Random forest classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
	   	- columns: vector-like object. The columns to use during classification. Defaults to all.
	   	## Returns
	   	- A matrix-like object consisting of performance metrics. 
	'''
	
	model = RandomForestClassifier(max_depth=2, random_state=0)

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return "{:.2}, {:.2}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

	# scores = cross_validate(model, data.to_numpy(), y.to_numpy(), cv=5, scoring=["accuracy", "recall"])

	# return "{:.2}, {:.2}".format(np.mean(scores['test_accuracy']), np.mean(scores['test_recall']))



def higradboost_classifier(data:pd.DataFrame, y:pd.Series, columns:pd.Series | list=None)->pd.DataFrame:
	'''
		Histogram gradient boosting classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
	   	- columns: vector-like object. The columns to use during classification. Defaults to all.
	   	## Returns
	   	- A matrix-like object consisting of performance metrics. 
	'''
	
	model = HistGradientBoostingClassifier()

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return "{:.2}, {:.2}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

	# scores = cross_validate(model, data.to_numpy(), y.to_numpy(), cv=5, scoring=["accuracy", "recall"])

	# return "{:.2}, {:.2}".format(np.mean(scores['test_accuracy']), np.mean(scores['test_recall']))



def lda_classifier(data:pd.DataFrame, y:pd.Series, columns:pd.Series | list=None)->pd.DataFrame:
	'''
		Linear discriminant analysis classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
	   	- columns: vector-like object. The columns to use during classification. Defaults to all.
	   	## Returns
	   	- A matrix-like object consisting of performance metrics. 
	'''
	
	model = LinearDiscriminantAnalysis()

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)

	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return "{:.2}, {:.2}".format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

	# scores = cross_validate(model, data.to_numpy(), y.to_numpy(), cv=5, scoring=["accuracy", "recall"])

	# return "{:.2}, {:.2}".format(np.mean(scores['test_accuracy']), np.mean(scores['test_recall']))





