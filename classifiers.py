'''
	algorithms.py

	Implementation of some ML classifiers.


	By Alph@B, AKA Brel MBE

'''


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import utils






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
		model = svm.SVC(kernel='linear', decision_function_shape='ovr', C=1)
	
	else:
		model = svm.SVC(kernel='linear', decision_function_shape='ovo', C=1)

	if columns != None:
		data = data[columns]

	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=0)


	# model.fit(X_train, y_train)
	# return model.score(X_test, y_test)

	scores = cross_val_score(model, data.to_numpy(), y.to_numpy(), cv=5)

	return scores.mean()



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


	# model.fit(X_train, y_train)
	# return model.score(X_test, y_test)

	scores = cross_val_score(model, data.to_numpy(), y.to_numpy(), cv=5)

	return scores.mean()



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


	# model.fit(X_train, y_train)
	# return model.score(X_test, y_test)

	scores = cross_val_score(model, data.to_numpy(), y.to_numpy(), cv=5)

	return scores.mean()



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


	# model.fit(X_train, y_train)
	# return model.score(X_test, y_test)

	scores = cross_val_score(model, data.to_numpy(), y.to_numpy(), cv=5)

	return scores.mean()



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


	# model.fit(X_train, y_train)
	# return model.score(X_test, y_test)

	scores = cross_val_score(model, data.to_numpy(), y.to_numpy(), cv=5)

	return scores.mean()



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


	# model.fit(X_train, y_train)
	# return model.score(X_test, y_test)

	scores = cross_val_score(model, data.to_numpy(), y.to_numpy(), cv=5)

	return scores.mean()





