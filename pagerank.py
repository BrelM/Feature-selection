'''
	pagerank.py

	Implement of pagerank algorithm for feature selection purposes.


	By Alph@B, AKA Brel MBE

'''



import numpy as np
import scipy.sparse
import networkx as nx




def pagerankloop(G:nx.Graph, columns:list, alpha:float=0.85, max_iter=None, pen_method="delete") -> np.array:
	'''
		Looping execution of the PageRank algorithm in other to select feautures based on the importance of their corresponding
		node in the provided graph.
		A penalization procedure is applied to the edges' weights or the personnalization vector to prevent two similar features to be selected.
		
		## Parameters
		G: NetworkX graph object.
			Weighted graph where M_i_j represents the information criteria between features i and j.
		columns: list.
			List of features.

		alpha: float.
			Damping factor, by default 0.85.
		max_iter: int.
			Maximum number of iterations.
		pen_method: str.
			Penalization method: delete, reduce.

		## Returns
		matrix-like object
			A vector of ranks such that v_i is the i-th rank from [0, 1].
	'''

	if max_iter is None:
		max_iter = G.number_of_nodes()

	features = []
	perso_vector = None

	for i in range(max_iter):
		v, perso_vector = pagerank(G, alpha, perso_vector)
		
		p = {b:perso_vector[a] for a, b in enumerate(G) }
		perso_vector = p

		pageranks = [(a, b) for b, a in v.items()]
		pageranks.sort()
		n = pageranks[-1][1]

		i = -1
		while columns[n] in features and len(pageranks) + i > 0:
			i -= 1
			n = pageranks[i][1]

		features.append(columns[n])

		# Updating features graph

		if pen_method == 'delete':
			
			# Strategy 1: Remove the selected node from the graph
			# along with the connected edges
			G.remove_node(n)
			perso_vector.pop(n)
			
		else:
			# Strategy 2: Penalize the weights of the edges connected to the selected node
			# along with the connected nodes and thus, the personalization vector
			edges_to_update = G.edges(columns[n], data=True)
			
			s = 0
			for edge in edges_to_update:
				perso_vector[edge[1]] -= edge[2]['weight']
				s += perso_vector[edge[1]] # Sum for normalization

			perso_vector[n] = 0

			for edge in edges_to_update:
				perso_vector[edge[1]] /= s

	return features





def pagerank(G, alpha=0.85, personalization=None, max_iter=20, tol=1.0e-6, weight='weight', dangling=None):
	'''
		PageRank algorithm with explicit number of iterations. Returns ranking of nodes (features) in
		the provided graph G.

		## Parameters
		G: NetworkX graph object.
			Weighted graph where M_i_j represents the information criteria between features i and j.
		p: numpy array.
			Personnalizatio vector for the PageRank algorithm.
		max_iter: int.
			Maximum number of iterations.

		tol: float.
			Tolerance factor (stopping criteria).
		
		alpha: float.
			Damping factor, by default 0.85.
		Dangling: bool.
			Rather to use dangling weights or not.

			
		## Returns
		numpy-array object
			A vector of ranks such that v_i is the i-th rank from [0, 1].
	'''
	
	N = len(G)
	if N == 0:
		return {}

	nodelist = list(G)
	M = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
	S = np.array(M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
	M = Q * M

	# initial vector
	x = np.repeat(1.0 / N, N)

	# Personalization vector
	if personalization is None:
		p = np.repeat(1.0 / N, N)
	else:
		p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
		p = p / p.sum()

	# Dangling nodes
	if dangling is None:
		dangling_weights = p
	else:
		# Convert the dangling dictionary into an array in nodelist order
		dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
		dangling_weights /= dangling_weights.sum()
	is_dangling = np.where(S == 0)[0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p

		# check convergence, l1 norm
		err = np.absolute(x - xlast).sum()
		if err < N * tol:
			# return list(map(float, x)), p
			return dict(zip(nodelist, map(float, x))), p
	
	# return list(map(float, x)), p
	return dict(zip(nodelist, map(float, x))), p  # raise nx.PowerIterationFailedConvergence(max_iter)
