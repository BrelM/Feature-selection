'''
	algorithms.py

	Implementation of some feature selection algorithms.


	By Alph@B, AKA Brel MBE

'''

import random

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn import svm


import utils


def relief(data:pd.DataFrame, y:pd.Series, m:int=10, n_features:int=5) -> list:
	'''
		Implementation of a reliefF family feature selection algorithm (binary classification).
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- m   :   interger. The number of times weights are updated.
		- n_features: integer. The desired number of selected features
	'''

	W = [0] * data.shape[1]

	for i in range(m):
		# print(f'Update {i + 1}...')
		a = random.randrange(data.shape[0])

		H, M = utils.hit_and_miss(data, y, a)

		for j in range(data.shape[1]):
			W[j] -= utils.diff(data, j, a, H) / m + utils.diff(data, j, a, M) / m
		# print(f"Features' weights W = \n{W}")
	

	# Returning selected features
	F = []
	for _ in range(n_features):
		m = np.argmin(W)
		F.append(list(data.columns)[m])

		W[m] = 1e9


	return F



def reliefF(data:pd.DataFrame, y:pd.Series, m:int=10, k:int=3, n_features:int=5) -> list:
	'''
		Implementation of a reliefF family feature selection algorithm (multiclass classification).
		## Parameters:		
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- m   :   interger. The number of times weights are updated.
		- k   :   interger. The number of nearest hits and misses to look for.
		- n_features: integer. The desired number of selected features
	'''

	W = [0] * data.shape[1]

	for i in range(m):
		# print(f'Update {i + 1}...')
		a = random.randrange(data.shape[0])

		H = utils.k_hits_or_misses(data, y, a, k)
		M = []
		for c in list(y.cat.categories):
			if c != y[a]:
				M.append(utils.k_hits_or_misses(data, y, a, k, c))
		

		for j in range(data.shape[1]):

			for _ in H:
				W[j] -= utils.diff(data, j, a, _) / (m * k)
			
			for c in M:
				temp_s = 0
				for _ in c:
					temp_s += utils.diff(data, j, a, _) / (m * k)
				
				W[j] += temp_s * (y.value_counts().get(y[c[0]], 0) / y.shape[0]) / (1 - (y.value_counts().get(y[a], 0) / y.shape[0]))
 

		# print(f"Features' weights W = \n{W}")


	# Returning selected features
	F = []
	for _ in range(n_features):
		m = np.argmin(W)
		F.append(list(data.columns)[m])

		W[m] = 1e9


	return F



def mutual_info(data:pd.DataFrame, y:pd.Series, n_features:int=5) -> list:
	'''
		Implementation of a mutual information-based feature selection algorithm (multiclass classification).
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- n_features: integer. The desired number of selected features
	'''

	# Feature selection
	test = SelectKBest(score_func=mutual_info_classif, k=n_features)
	fit = test.fit(data.to_numpy(), y.to_numpy())

	# Summarize scores
	np.set_printoptions(precision=3)
	# print(fit.scores_)

	# features = fit.transform(data.to_numpy())


	# Returning selected features
	F = []
	scores = list(fit.scores_)
	for _ in range(n_features):
		m = np.argmin(scores)
		F.append(list(data.columns)[m])

		scores[m] = 1e9


	return F



def forward_FS(data:pd.DataFrame, y:pd.Series, n_features:int=5) -> list:
	'''
		Implementation of a sequential features selection algorithm (forward version) using logistic regression as a classifier.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- n_features: integer. The desired number of selected features
	'''

	model = LogisticRegression()

	# Feature selection
	fs = SequentialFeatureSelector(model, n_features_to_select=n_features, scoring='accuracy')

	fs.fit(data.to_numpy(), y.to_numpy())


	features = fs.get_support()


	# Returning selected features
	F = list(data.columns[features])

	return F



def ridge_fs(data:pd.DataFrame, y:pd.Series, alpha:int=5, n_features:int=5) -> list:
	'''
		Implementation of a ridge regression-based features selection algorithm.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- alpha: integer. The regularization coefficient.
		- n_features: integer. The desired number of selected features

	'''

	# Fitting the feature selection the model
	ridgereg = Ridge(alpha=alpha)
	ridgereg.fit(data.to_numpy(), y.to_numpy())


	# Returning selected features
	F = []
	scores = list(ridgereg.coef_)
	for _ in range(n_features):
		m = np.argmin(scores)
		F.append(list(data.columns)[m])

		scores[m] = 1e9


	return F



def lasso_fs(data:pd.DataFrame, y:pd.Series, alpha:int=1e-10, n_features:int=5) -> list:
	'''
		Implementation of a Lasso regression-based features selection algorithm.
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- alpha: integer. The regularization coefficient.
		- n_features: integer. The desired number of selected features

	'''

	# Fitting the feature selection the model
	lassoreg = Lasso(alpha=alpha, max_iter=int(1e5))
	lassoreg.fit(data.to_numpy(), y.to_numpy())


	# Returning selected features
	F = []
	scores = list(lassoreg.coef_)
	for _ in range(n_features):
		m = np.argmin(scores)
		F.append(list(data.columns)[m])

		scores[m] = 1e9


	return F



def svm_rfe(data:pd.DataFrame, y:pd.Series, n_features:int=5) -> list:
	'''
		Implementation of a suport vector machine sequential features selection algorithm (backward elimination version).
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- n_features: integer. The desired number of selected features
	'''

	if y.cat.categories.shape[0] > 2:
		while data.shape[1] > n_features:
		
			# Creating and fitting the SVM model over the data
			model = svm.SVC(kernel='linear', decision_function_shape='ovr')

			# print(f'\nNo. of columns: {data.columns.shape[0]}\nAvailable columns: {list(data.columns)}')
			model.fit(data.to_numpy(), y.to_numpy())

			# Removing the feature with the less contribution
			coef_means = model.coef_.mean(axis=0)
			a = data.columns[np.argmin(coef_means)]
			
			# print(f"Removing {a}")

			data.drop(a, axis='columns', inplace=True)
			# print(f'Dropped column : {a}')	

	else:
		while data.shape[1] > n_features:
			
			# Creating and fitting the SVM model over the data
			model = svm.SVC(kernel='linear')

			# print(f'\nNo. of columns: {data.columns.shape[0]}\nAvailable columns: {list(data.columns)}')
			model.fit(data.to_numpy(), y.to_numpy())

			# Removing the feature with the less contribution
			a = data.columns[np.argmin(model.coef_)]
			
			# print(f"Removing {a}")

			data.drop(a, axis='columns', inplace=True)
			# print(f'Dropped column : {a}')
	
	return list(data.columns)



def svm_rfe_sfs(data:pd.DataFrame, y:pd.Series, n_features:int=5) -> list:
	'''
		Implementation of a suport vector machine sequential features selection algorithm (backward elimination version).
		## Parameters:
		- data:   dateframe-like object. The data on hich the processing is done.
		- y   :   vector-like object. The class labels of the data.
		- n_features: integer. The desired number of selected features
	'''

	if y.cat.categories.shape[0] > 2:
		model = svm.SVC(kernel='linear', decision_function_shape='ovr')
	
	else:
		model = svm.SVC(kernel='linear', decision_function_shape='ovo')

	# Feature selection
	fs = SequentialFeatureSelector(model, n_features_to_select=n_features, scoring='accuracy')

	fs.fit(data.to_numpy(), y.to_numpy())


	features = fs.get_support()


	# Returning selected features
	F = list(data.columns[features])

	return F



