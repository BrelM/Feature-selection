"""By Arielle Kana"""

import numpy as np


"""
Generic PageRank Solver

Formule générale :
PR = (1 - d) * v + d * P^T * PR

Paramètres :
- W : matrice de poids (graphe)
- v : vecteur de préférence (si None → uniforme)
- damping : facteur d'amortissement
- max_iter : nombre max d'itérations
- tol : critère de convergence
"""


def pagerank(W, v=None, damping=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank scores.

    Parameters
    ----------
    W : adjacency matrix
    v : personalization vector
    damping : damping factor

    Returns
    -------
    scores : PageRank scores
    """

    n = W.shape[0]

    # normalisation des lignes
    row_sum = W.sum(axis=1)

    row_sum[row_sum == 0] = 1

    P = W / row_sum[:, None]

    # vecteur personnalisation
    if v is None:
        v = np.ones(n) / n
    else:
        v = v / np.sum(v)

    scores = np.ones(n) / n

    for _ in range(max_iter):

        new_scores = damping * P.T @ scores + (1 - damping) * v

        if np.linalg.norm(new_scores - scores) < tol:
            break

        scores = new_scores

    return scores


def build_weighted_graph(S, threshold=0):
    """
    onstruit un graphe pondéré des features à partir
    d'une matrice de similarité.

    Paramètres
    ----------
    similarity_matrix : ndarray (n_features, n_features)
        Similarité entre paires de features

    threshold : float
        Seuil pour éliminer les connexions faibles

    Retour
    ------
    W : ndarray (n_features, n_features)
        Matrice d'adjacence pondérée

    Build weighted adjacency matrix.

    Parameters
    ----------
    S : similarity matrix
    threshold : float

    Returns
    -------
    W : adjacency matrix
    """

    W = S.copy()

    W[W < threshold] = 0

    # supprimer auto-connexions
    np.fill_diagonal(W, 0)

    return W