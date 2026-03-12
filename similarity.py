"""By Arielle Kana"""

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mutual_info_score


def cosine_similarity_matrix(X):
    """
    Calcule la matrice de similarité cosinus entre features.

    Chaque feature est vue comme un vecteur de dimension n_samples.
    La similarité cosinus mesure l'angle entre deux features.

    Compute cosine similarity between features.

    Parameters
    ----------
    X : ndarray (samples × features)

    Returns
    -------
    S : ndarray (features × features)
        Cosine similarity matrix.
    """

    # normalisation des colonnes
    norm = np.linalg.norm(X, axis=0)

    norm[norm == 0] = 1e-12

    X_norm = X / norm

    # produit matriciel
    S = X_norm.T @ X_norm

    return S


def pearson_similarity_matrix(X):
    """
    Compute Pearson correlation similarity between features.
    """

    n_features = X.shape[1]

    S = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):

            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]

            if np.isnan(corr):
                corr = 0

            S[i, j] = abs(corr)

    return S


def kendall_similarity_matrix(X):
    """
    Similarité basée sur le coefficient de Kendall tau.
    Utile pour Learning to Rank.
    """
    n_features = X.shape[1]
    S = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            tau, _ = kendalltau(X[:, i], X[:, j])
            S[i, j] = S[j, i] = 0.0 if np.isnan(tau) else abs(tau)

    return S


def mutual_information_matrix(X, n_bins=10):
    """
    Calcule une matrice d'information mutuelle entre features.
    Discrétisation simple par histogramme.
    """
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    # Discrétisation
    X_discrete = np.zeros_like(X)
    for i in range(n_features):
        X_discrete[:, i] = np.digitize(X[:, i], bins=np.histogram(X[:, i], bins=n_bins)[1][:-1])

    for i in range(n_features):
        for j in range(n_features):
            mi_matrix[i, j] = mutual_info_score(X_discrete[:, i], X_discrete[:, j])

    return mi_matrix
