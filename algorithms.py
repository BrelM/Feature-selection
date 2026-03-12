'''
	algorithms.py

	Implementation of some feature selection algorithms.


	By Alph@B, AKA Brel MBE & Arielle Kana

'''

import random

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn import svm



from similarity import( 
 	cosine_similarity_matrix,
 	pearson_similarity_matrix)

from graph_utils import (
    pagerank,
    build_weighted_graph)

from information_theory import (
    mutual_information_with_target,
	mutual_information_matrix,
	entropy_vector)



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


def select_top_features(scores, columns, n_features):
    """
    Select the top features according to their scores.

    Parameters
    ----------
    scores : array-like
        Score of each feature.
    columns : array-like
        Feature names.
    n_features : int or float
        Number of features or ratio.

    Returns
    -------
    list
        Selected feature names.
    """

    n = len(columns)

    # Si n_features est un ratio
    if isinstance(n_features, float):
        k = max(1, int(n_features * n))
    else:
        k = min(n_features, n)

    # indices triés par score décroissant
    indices = np.argsort(scores)[::-1][:k]

    return list(columns[indices])



def ugfs(data: pd.DataFrame, y: pd.Series=None, n_features=5, damping=0.85):
    """
    UGFS - Unsupervised Graph-based Feature Selection

    Steps:
    1. Build feature similarity graph
    2. Compute PageRank centrality
    3. Select top-k features
    """

    X = data.to_numpy()
    columns = data.columns.to_numpy()

    # ---- STEP 1: construire graphe de similarité entre features ----
    # Similarité cosine entre features
    S = cosine_similarity_matrix(X)

    S = np.nan_to_num(S)

    # Matrice d'adjacence pondérée
    W = build_weighted_graph(S)

    # ---- STEP 2: calcul PageRank ----
    scores = pagerank(W, damping=damping)

    scores = np.array(scores)

    # ---- STEP 3: sélectionner top features ----
    return select_top_features(scores, columns, n_features)


def pprfs(data: pd.DataFrame, y: pd.Series, n_features=5, damping=0.85, beta=0.7):
    """
    PPRFS - Personalized PageRank Feature Selection

    Implementation fidèle à l'article :

    1. Construire réseau de redondance des features
    2. Sélection gloutonne
    3. Personalized PageRank recalculé à chaque itération
    """

    X = data.to_numpy()
    columns = data.columns.to_numpy()

    m = X.shape[1]

    # ---- STEP 1: construire matrice de redondance ----
    MI = mutual_information_matrix(X)

    H = entropy_vector(X)

    R = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if H[j] > 0:
                R[i, j] = MI[i, j] / H[j]

    W = build_weighted_graph(R)

    # ---- STEP 2: calcul pertinence I(fi ; C) ----
    relevance = mutual_information_with_target(X, y)

    selected = []

    candidate_indices = list(range(m))

    while len(selected) < n_features:

        # ---- construire vecteur personnalisation ----
        v = np.zeros(m)

        if len(selected) == 0:
            v[:] = 1 / m
        else:
            for idx in selected:
                v[idx] = 1 / len(selected)

        # ---- PageRank personnalisé ----
        pr = pagerank(W, v=v, damping=damping)

        pr = np.array(pr)

        # ---- score final ----
        scores = relevance - beta * pr

        # ignorer features déjà sélectionnées
        scores[selected] = -np.inf

        best = np.argmax(scores)

        selected.append(best)

    return list(columns[selected])


def mgfs(data: pd.DataFrame, y: pd.Series=None, n_features=5, damping=0.85):
    """
    MGFS - Multi Graph Feature Selection

    Combine plusieurs graphes de similarité.
    """

    X = data.to_numpy()
    columns = data.columns.to_numpy()

    # ---- construire plusieurs graphes ----
    S1 = cosine_similarity_matrix(X)
    S2 = pearson_similarity_matrix(X)
    S3 = mutual_information_matrix(X)

    S1 = np.nan_to_num(S1)
    S2 = np.nan_to_num(S2)
    S3 = np.nan_to_num(S3)

    W1 = build_weighted_graph(S1)
    W2 = build_weighted_graph(S2)
    W3 = build_weighted_graph(S3)

    # ---- PageRank pour chaque graphe ----
    pr1 = np.array(pagerank(W1, damping=damping))
    pr2 = np.array(pagerank(W2, damping=damping))
    pr3 = np.array(pagerank(W3, damping=damping))

    # ---- fusion des scores ----
    scores = (pr1 + pr2 + pr3) / 3

    return select_top_features(scores, columns, n_features)



def sgfs(data: pd.DataFrame, y: pd.Series, n_features=5, damping=0.85):
    """
    SGFS - Semi-supervised Graph Feature Selection
	Principe :

	utiliser information du label

	pour guider PageRank
    """

    X = data.to_numpy()
    y_array = y.to_numpy()
    columns = data.columns.to_numpy()

    # ---- graphe entre features ----
    S = cosine_similarity_matrix(X)
    S = np.nan_to_num(S)

    W = build_weighted_graph(S)

    # ---- corrélation avec label ----
    correlations = []

    for i in range(X.shape[1]):

        corr = np.corrcoef(X[:, i], y_array)[0, 1]

        if np.isnan(corr):
            corr = 0

        correlations.append(abs(corr))

    correlations = np.array(correlations)

    # vecteur personnalisation
    v = correlations / (np.sum(correlations) + 1e-12)

    # PageRank supervisé
    scores = pagerank(W, v=v, damping=damping)

    scores = np.array(scores)

    return select_top_features(scores, columns, n_features)


def fss_cpr(data: pd.DataFrame, y: pd.Series, n_features=5, damping=0.85, alpha=0.5):
    """
    FSS-CPR - Feature Selection via Collaborative PageRank

	Principe :

	combiner

	centralité du graphe

	pertinence supervisée
    """

    X = data.to_numpy()
    y_array = y.to_numpy()
    columns = data.columns.to_numpy()

    # ---- graphe entre features ----
    S = cosine_similarity_matrix(X)
    S = np.nan_to_num(S)

    W = build_weighted_graph(S)

    # ---- PageRank ----
    pr_scores = np.array(pagerank(W, damping=damping))

    # ---- pertinence supervisée ----
    sup_scores = []

    for i in range(X.shape[1]):

        corr = np.corrcoef(X[:, i], y_array)[0, 1]

        if np.isnan(corr):
            corr = 0

        sup_scores.append(abs(corr))

    sup_scores = np.array(sup_scores)

    # ---- fusion ----
    scores = alpha * pr_scores + (1 - alpha) * sup_scores

    return select_top_features(scores, columns, n_features)