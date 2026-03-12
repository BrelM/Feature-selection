"""By Arielle Kana"""

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
import numpy as np



def entropy_vector(X, bins=10):
    """
    Compute entropy of each feature.

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    H : array
        entropy for each feature
    """

    n_features = X.shape[1]

    H = []

    for i in range(n_features):

        hist, _ = np.histogram(X[:, i], bins=bins)

        p = hist / np.sum(hist)

        p = p[p > 0]

        entropy = -np.sum(p * np.log2(p))

        H.append(entropy)

    return np.array(H)



def mutual_information_matrix(X):
    """
    Calcule la matrice de similarité cosinus entre features.

    Chaque feature est vue comme un vecteur de dimension n_samples.
    La similarité cosinus mesure l'angle entre deux features.
    
    Compute mutual information between all feature pairs.
    """

    n_features = X.shape[1]

    MI = np.zeros((n_features, n_features))

    for i in range(n_features):

        mi = mutual_info_regression(X, X[:, i])

        MI[i] = mi

    return MI

    return np.array(H)

def mutual_information_with_target(X, y):
    """
    Compute mutual information between features and target.
    """

    mi = mutual_info_classif(X, y)

    mi = np.nan_to_num(mi)

    return mi