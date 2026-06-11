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



def lasso_fs(data:pd.DataFrame, y:pd.Series, alpha:float=1e-10, n_features:int=5) -> list:
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

'''
algorithms.py

Graph-based feature selection algorithms — implementations faithful to
the source papers.

Papers covered
--------------
  UGFS    : Henni et al., Expert Syst. Appl. 114 (2018) 46-53
  PPRFS   : Zhu et al., IEEE 2019
  MGFS    : Hashemi et al., Expert Syst. Appl. 142 (2020) 113024
  SGFS    : Dalvand et al., CSICC 2022
  FSS-CPR : Yeh & Tsai, ICCIP 2021  (FS-SCPR, adapted for general FS)

All algorithms:
  • encode categorical features automatically (LabelEncoder)
  • accept a `damping` parameter iterated over [0.15, 0.50, 0.85] by worker.py

By Alph@B, AKA Brel MBE & Arielle Kana
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _encode_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *data* with all non-numeric columns
    label-encoded as float64.

    Handles pandas CategoricalDtype columns correctly.
    """
    out = data.copy()
    for col in out.columns:
        col_dtype = out[col].dtype
        is_numeric = False
        try:
            is_numeric = np.issubdtype(col_dtype, np.number)
        except TypeError:
            is_numeric = False  # CategoricalDtype raises TypeError
        if not is_numeric:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str)).astype(float)
    return out.astype(float)


def _encode_y(y) -> np.ndarray:
    """
    Convert *y* (Series, array, or list) to a 1-D float numpy array,
    label-encoding string/categorical labels when necessary.

    Handles pandas CategoricalDtype, object dtype, and any non-numeric type.
    """
    # Extract underlying values from pandas objects
    if hasattr(y, 'to_numpy'):
        # Use to_numpy() to get a plain ndarray, stripping pandas metadata
        arr = y.to_numpy()
    elif hasattr(y, 'values'):
        arr = y.values
    else:
        arr = np.array(y)

    # Convert to float directly when possible
    try:
        return arr.astype(float)
    except (ValueError, TypeError):
        # Fallback: label-encode anything that cannot be cast to float
        le = LabelEncoder()
        return le.fit_transform(arr.astype(str)).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE-SELECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def select_top_features(scores, columns, n_features):
    """
    Return the names of the top-k features sorted by *scores* (descending).

    Parameters
    ----------
    scores     : array-like of shape (m,)
    columns    : array-like of shape (m,)  — feature names
    n_features : int | float  — count or ratio
    """
    n = len(columns)
    if isinstance(n_features, float):
        k = max(1, int(n_features * n))
    else:
        k = min(int(n_features), n)
    indices = np.argsort(scores)[::-1][:k]
    return list(np.array(columns)[indices])


# ─────────────────────────────────────────────────────────────────────────────
# INFORMATION-THEORY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _entropy(x: np.ndarray) -> float:
    """Marginal entropy H(X) estimated with equal-width histogram bins."""
    n = len(x)
    bins = max(2, int(np.sqrt(n)))
    counts, _ = np.histogram(x, bins=bins)
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log(p + 1e-12)))


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information I(X;Y) estimated with a 2-D histogram."""
    n = len(x)
    bins = max(2, int(np.sqrt(n)))
    h2, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = h2 / n
    px  = pxy.sum(axis=1, keepdims=True)
    py  = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / (px * py + 1e-12)[mask]))
    return float(max(0.0, mi))


def _mi_matrix(X: np.ndarray) -> np.ndarray:
    """Symmetric pairwise MI matrix M where M[i,j] = I(Xi ; Xj)."""
    m = X.shape[1]
    M = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            v = _mutual_info(X[:, i], X[:, j])
            M[i, j] = M[j, i] = v
    return M


def _entropy_vector(X: np.ndarray) -> np.ndarray:
    """Return H(Xj) for every feature column j."""
    return np.array([_entropy(X[:, j]) for j in range(X.shape[1])])


def _mi_with_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return I(Xj ; Y) for every feature column j."""
    return np.array([_mutual_info(X[:, j], y) for j in range(X.shape[1])])


def _correlation_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Correlation distance CD(x, y) = 1 – Pearson_corr(x, y).
    Used by MGFS and SGFS.  Range: [0, 2].
    """
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 1.0
    r = float(np.corrcoef(x, y)[0, 1])
    return 1.0 - (r if not np.isnan(r) else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE-RANK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def pagerank(W: np.ndarray,
             damping: float = 0.85,
             v: np.ndarray = None,
             max_iter: int = 200,
             tol: float = 1e-6) -> np.ndarray:
    """
    Power-iteration PageRank on a weighted adjacency matrix W.

    W[i, j] represents the weight of the edge i → j.
    Handles dangling nodes (rows summing to zero).
    When *v* is provided the computation becomes *Personalized* PageRank
    (Eq. 9–10 in Zhu et al. 2019; Eq. 3 in Henni et al. 2018).

    Parameters
    ----------
    W       : (m, m) non-negative weight matrix
    damping : damping factor α  (typically 0.85)
    v       : personalisation / teleportation vector (uniform if None)
    """
    n = W.shape[0]

    # Transition matrix P (row-stochastic)
    row_sums  = W.sum(axis=1)
    dangling  = (row_sums == 0)
    P = np.zeros_like(W, dtype=float)
    for i in range(n):
        if row_sums[i] > 0:
            P[i] = W[i] / row_sums[i]

    if v is None:
        v = np.ones(n) / n
    else:
        v = np.asarray(v, dtype=float)
        s = v.sum()
        v = v / s if s > 1e-12 else np.ones(n) / n

    pr = np.ones(n) / n
    for _ in range(max_iter):
        dangling_contrib = damping * float(pr[dangling].sum()) * v
        pr_new = damping * (P.T @ pr) + dangling_contrib + (1.0 - damping) * v
        pr_new /= pr_new.sum() + 1e-12
        if np.linalg.norm(pr_new - pr, 1) < tol:
            pr = pr_new
            break
        pr = pr_new
    return pr


# ─────────────────────────────────────────────────────────────────────────────
# UGFS — Henni et al., Expert Syst. Appl. 114 (2018) 46-53
# ─────────────────────────────────────────────────────────────────────────────

def ugfs(data: pd.DataFrame,
         y: pd.Series = None,
         n_features: int = 5,
         damping: float = 0.85,
         k_neighbors: int = 5,
         delta: float = None) -> list:
    """
    Unsupervised Graph-based Feature Selection (UGFS).

    Algorithm (faithful to Algorithm 1, Henni et al. 2018)
    -------------------------------------------------------
    1. For every data point xp, compute its k-NN neighbourhood NNk(xp).
    2. Compute var_j(NNk(xp)) for each feature j in the neighbourhood.
    3. Subspace preference cluster Sp = {j : var_j(NNk(xp)) ≤ δ}.
    4. Set A[fi, fj] = 1 for all (fi, fj) ∈ Sp  (binary adjacency).
    5. Apply PageRank on the resulting undirected graph.
    6. Return the top-k features.

    Categorical variables are label-encoded before processing.
    δ is auto-estimated as the median neighbourhood variance when None.
    """
    # --- preprocessing -------------------------------------------------------
    data_enc = _encode_df(data)
    X = data_enc.to_numpy()
    columns = data.columns.to_numpy()
    n, m = X.shape

    # Z-score normalisation for meaningful variance comparisons
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Xn = (X - mu) / sigma

    # --- k-NN ----------------------------------------------------------------
    k = min(k_neighbors, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(Xn)
    _, nn_idx = nbrs.kneighbors(Xn)

    # --- auto-estimate δ as median neighbourhood variance -------------------
    if delta is None:
        all_vars = [float(np.var(Xn[nn_idx[i], j]))
                    for i in range(n) for j in range(m)]
        delta = float(np.median(all_vars))

    # --- build binary adjacency matrix (Algorithm 1 of the paper) -----------
    A = np.zeros((m, m), dtype=float)
    for i in range(n):
        Sp = [j for j in range(m)
              if float(np.var(Xn[nn_idx[i], j])) <= delta]
        for fi in Sp:
            for fj in Sp:
                A[fi, fj] = 1.0
    np.fill_diagonal(A, 0.0)

    scores = pagerank(A, damping=damping)
    return select_top_features(scores, columns, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# PPRFS — Zhu et al., IEEE 2019
# ─────────────────────────────────────────────────────────────────────────────

def pprfs(data: pd.DataFrame,
          y: pd.Series,
          n_features: int = 5,
          damping: float = 0.85,
          beta: float = 0.7) -> list:
    """
    Personalized PageRank-based Feature Selection (PPRFS).

    Algorithm (faithful to Zhu et al. 2019)
    ----------------------------------------
    Feature redundancy network (directed graph)
        Edge fj → fi with weight  W[j,i] = R(fi;fj) = I(fi;fj) / H(fj)
    
    Greedy selection loop:
        1. Compute Personalised PageRank with teleportation toward S.
        2. Normalise candidate PR values (Eq. 11).
        3. Select fi = argmax  I(fi ; C) − β · pi'   (Eq. 12).
        4. Add fi to S; update teleportation vector.

    Categorical variables are label-encoded before processing.
    """
    # --- preprocessing -------------------------------------------------------
    data_enc = _encode_df(data)
    X = data_enc.to_numpy()
    y_arr = _encode_y(y)
    columns = data.columns.to_numpy()
    m = X.shape[1]

    # --- redundancy network --------------------------------------------------
    MI = _mi_matrix(X)
    H  = _entropy_vector(X)

    # R[i,j] = I(fi;fj)/H(fj)  — directed edge fj→fi has weight R[i,j]
    # In our W matrix W[j,i] = R[i,j], i.e. W = R.T  (edge from-row to-col)
    R = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if H[j] > 1e-12:
                R[i, j] = MI[i, j] / H[j]

    # Directed adjacency: W[j,i] = R(fi;fj)
    W = R.T.copy()
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, 0.0)

    # --- feature relevance ---------------------------------------------------
    relevance = _mi_with_target(X, y_arr)

    # --- greedy selection ----------------------------------------------------
    n_sel = int(n_features)
    selected: list[int] = []
    v = np.ones(m) / m          # uniform teleportation for S = ∅

    while len(selected) < n_sel:
        pr = pagerank(W, v=v, damping=damping)

        # candidates
        c_idx = np.array([i for i in range(m) if i not in selected])
        if len(c_idx) == 0:
            break

        # Normalise PR of candidates so Σ pi' (candidates) = Σ I(fi;C) (candidates)
        # (Eq. 11 of the paper)
        sum_rel_c = relevance[c_idx].sum()
        sum_pr_c  = pr[c_idx].sum()
        pr_norm = pr.copy()
        if sum_pr_c > 1e-12:
            pr_norm[c_idx] *= sum_rel_c / sum_pr_c

        # Selection criterion (Eq. 12)
        score = relevance - beta * pr_norm
        score[selected] = -np.inf

        best = int(np.argmax(score))
        selected.append(best)

        # Update teleportation vector: uniform over selected set
        v = np.zeros(m)
        for idx in selected:
            v[idx] = 1.0 / len(selected)

    return list(columns[selected])


# ─────────────────────────────────────────────────────────────────────────────
# MGFS — Hashemi et al., Expert Syst. Appl. 142 (2020) 113024
# ─────────────────────────────────────────────────────────────────────────────

def mgfs(data: pd.DataFrame,
         y: pd.Series = None,
         n_features: int = 5,
         damping: float = 0.85) -> list:
    """
    Multi-label Graph-based Feature Selection via PageRank centrality (MGFS).

    Algorithm (faithful to Hashemi et al. 2020)
    --------------------------------------------
    1. Correlation Distance Matrix  CDM[i, l] = 1 − corr(Xi, Yl)
       (m × L matrix; L = number of labels; adapted to L=1 for single-label).
    2. Euclidean Distance Matrix  EDM[i, j] = ||CDM[i] − CDM[j]||₂
       (m × m, encodes how differently each feature correlates with labels).
    3. Build complete weighted Feature-Label Graph (FLG) using EDM as weights.
    4. Apply Weighted PageRank — features with unique label correlations
       (large EDM distances from others) receive higher scores.
    5. Return top-k features.

    Categorical variables are label-encoded before processing.
    """
    # --- preprocessing -------------------------------------------------------
    data_enc = _encode_df(data)
    X = data_enc.to_numpy()
    columns = data.columns.to_numpy()
    m = X.shape[1]

    if y is not None:
        if isinstance(y, pd.DataFrame):
            Y = y.values.astype(float)
        else:
            Y = _encode_y(y).reshape(-1, 1)
    else:
        Y = X.copy()          # unsupervised fallback

    L = Y.shape[1] if Y.ndim > 1 else 1
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # --- Step 1: CDM (m × L) -------------------------------------------------
    CDM = np.zeros((m, L))
    for i in range(m):
        for j in range(L):
            CDM[i, j] = _correlation_distance(X[:, i], Y[:, j])

    # --- Step 2: EDM (m × m) -------------------------------------------------
    EDM = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            diff = CDM[i] - CDM[j]
            EDM[i, j] = float(np.sqrt(np.dot(diff, diff)))

    # --- Step 3-4: Weighted PageRank on FLG ----------------------------------
    W = EDM.copy()
    np.fill_diagonal(W, 0.0)

    scores = pagerank(W, damping=damping)
    return select_top_features(scores, columns, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# SGFS — Dalvand et al., CSICC 2022
# ─────────────────────────────────────────────────────────────────────────────

def sgfs(data: pd.DataFrame,
         y: pd.Series,
         n_features: int = 5,
         damping: float = 0.85) -> list:
    """
    Semi-supervised Graph-based Feature Selection (SGFS).

    Algorithm (faithful to Dalvand et al. 2022, Algorithm 1)
    ---------------------------------------------------------
    1. Rel_CDM[i]   = CD(Xi, Y)           feature-label correlation distance
    2. Red_CDM[i,j] = CD(Xi, Xj)          feature-feature correlation distance
    3. CDM = [Rel_CDM | mean_j(Red_CDM)]  combine relevance & redundancy (m×2)
    4. EDM[i,j] = ||CDM[i] − CDM[j]||₂   Euclidean distance in CDM space
    5. Weighted PageRank on complete graph with EDM weights → feature scores
    6. Return top-k features.

    Categorical variables are label-encoded before processing.
    """
    # --- preprocessing -------------------------------------------------------
    data_enc = _encode_df(data)
    X = data_enc.to_numpy()
    y_arr = _encode_y(y)
    columns = data.columns.to_numpy()
    m = X.shape[1]

    # --- Step 1: Relevance — Rel_CDM (m × 1) --------------------------------
    Rel_CDM = np.array([_correlation_distance(X[:, i], y_arr)
                        for i in range(m)]).reshape(m, 1)

    # --- Step 2: Redundancy — Red_CDM (m × m) --------------------------------
    Red_CDM = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                Red_CDM[i, j] = _correlation_distance(X[:, i], X[:, j])

    # --- Step 3: Combined CDM (m × 2) ----------------------------------------
    mean_Red = Red_CDM.mean(axis=1, keepdims=True)  # avg redundancy per feature
    CDM = np.hstack([Rel_CDM, mean_Red])

    # --- Step 4: EDM (m × m) -------------------------------------------------
    EDM = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            diff = CDM[i] - CDM[j]
            EDM[i, j] = float(np.sqrt(np.dot(diff, diff)))

    # --- Step 5: Weighted PageRank -------------------------------------------
    W = EDM.copy()
    np.fill_diagonal(W, 0.0)

    scores = pagerank(W, damping=damping)
    return select_top_features(scores, columns, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# FSS-CPR — Yeh & Tsai, ICCIP 2021  (FS-SCPR, adapted for general FS)
# ─────────────────────────────────────────────────────────────────────────────

def fss_cpr(data: pd.DataFrame,
            y: pd.Series,
            n_features: int = 5,
            damping: float = 0.85,
            sigma: float = 0.1) -> list:
    """
    Feature Selection via Spectral Clustering + biased PageRank (FSS-CPR).

    Adapted from Yeh & Tsai (2021) FS-SCPR for general feature selection
    (original context was Learning to Rank with query-document pairs).

    Steps
    -----
    1. Build undirected feature similarity graph W where
       W[i,j] = |corr(Xi,Xj)|  if |corr| ≥ σ, else 0.
       (Pearson correlation used as proxy for Kendall's τ on ranking lists.)
    2. Spectral Clustering → partition features into k clusters,
       aiming for minimum within-cluster redundancy.
    3. Biased PageRank with preference p(fi) ∝ |corr(Xi, Y)|
       (MAP score in the original paper → feature-label correlation here).
    4. For each cluster, select the representative feature that maximises
       0.5 × PR_score  +  0.5 × normalised within-cluster similarity.
    5. If fewer representatives than n_features, fill with highest-PR features.

    Categorical variables are label-encoded before processing.
    """
    # --- preprocessing -------------------------------------------------------
    data_enc = _encode_df(data)
    X = data_enc.to_numpy()
    y_arr = _encode_y(y)
    columns = data.columns.to_numpy()
    m = X.shape[1]
    n_sel = int(n_features)

    # --- Step 1: Feature similarity graph (Eq. 2 of the paper) --------------
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if np.isnan(r):
                r = 0.0
            sim = abs(r)
            if sim >= sigma:
                W[i, j] = W[j, i] = sim

    # --- Step 2: Spectral Clustering -----------------------------------------
    n_clusters = max(2, min(n_sel, max(2, m // 2)))
    affinity = W + np.eye(m) * 1e-6   # ensure positive semi-definiteness
    try:
        sc = SpectralClustering(n_clusters=n_clusters,
                                affinity='precomputed',
                                random_state=42,
                                n_init=10)
        labels = sc.fit_predict(affinity)
    except Exception:
        labels = np.arange(m) % n_clusters   # graceful fallback

    # --- Step 3: Biased PageRank (Eq. 3 of the paper) -----------------------
    pref = np.array([
        abs(float(np.corrcoef(X[:, i], y_arr)[0, 1]))
        if not np.isnan(np.corrcoef(X[:, i], y_arr)[0, 1]) else 0.0
        for i in range(m)
    ])
    pref_sum = pref.sum()
    pref = pref / pref_sum if pref_sum > 1e-12 else np.ones(m) / m
    pr = pagerank(W, v=pref, damping=damping)

    # --- Step 4: Representative per cluster (Step 2 of Section 3.4) ---------
    selected: list[int] = []
    for c in range(n_clusters):
        c_idx = np.where(labels == c)[0]
        if len(c_idx) == 0:
            continue
        # SSim(f) = sum of pairwise similarities within cluster (normalised)
        ssim = np.array([
            W[fi, c_idx].sum() - W[fi, fi]
            for fi in c_idx
        ])
        ssim /= max(1, len(c_idx) - 1)
        # Combined selection score (Step 2.2)
        combined = 0.5 * pr[c_idx] + 0.5 * ssim
        best = int(c_idx[np.argmax(combined)])
        selected.append(best)

    # --- Step 5: top-up if needed --------------------------------------------
    if len(selected) < n_sel:
        already = set(selected)
        extras = sorted(
            [(i, pr[i]) for i in range(m) if i not in already],
            key=lambda t: t[1], reverse=True
        )
        for idx, _ in extras[:n_sel - len(selected)]:
            selected.append(idx)

    return list(columns[selected[:n_sel]])


