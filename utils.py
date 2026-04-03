'''
	utils.py

	Some useful functions and procedures.


	By Alph@B, AKA Brel MBE

'''



import pandas as pd
import numpy as np
from scipy.stats import f_oneway, chi2_contingency
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from feature_engine.discretisation import DecisionTreeDiscretiser



def load_data(data_params:dict, encode_cat:bool=True) -> tuple[pd.DataFrame, pd.Series]:
	'''
		Load data at path using data_params and return inputs as X and targets as Y.
		## Parameters:
		- data_params: A dictionnary containing:
			- path: path to the data file
			- nb_features: number of features
			- class_idx: the class column index
			- sep: the file separator
			- id: If there is an Id column or not
			- labels: If there are labels in the file
		- encode_cat: bool.
					  Rather to encode categorical attributes or not.

	'''
	
	if data_params['labels']:
		data = pd.read_csv(data_params["path"], sep=data_params['sep'], engine='python')
		data.rename(columns={a:b for a, b in zip(data.columns, ['A' + str(i) for i in range(data_params['nb_features'])])}, inplace=True)
	else:
		data = pd.read_csv(data_params["path"], sep=data_params['sep'], engine='python', names=['A' + str(i) for i in range(data_params['nb_features'])])


	# If there is an ID column, we remove it
	if data_params['id']:
		data.drop('A0', axis='columns', inplace=True)
	
	
	class_label = 'A' + str(data_params['class_idx'])
	y = data[class_label].astype('category')
	data.drop(class_label, axis='columns', inplace=True)
	n = y.cat.categories.shape[0]
	
	
	# Seperating the categorical features from the numerical ones
	categorical_columns = data.select_dtypes(include=['object', 'category']).columns
	numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
	
	# Dealing with missing values
	num_cols = data[numerical_columns]
	for col in numerical_columns:
		mean_value = data[col].mean()
		num_cols.loc[:, col] = data.loc[:, col].fillna(mean_value)


	cat_cols = data[categorical_columns]
	for col in categorical_columns:
		mode_value = data[col].mode()[0]
		cat_cols.loc[:, col] = data.loc[:, col].fillna(mode_value)


	# Scaling numerical values
	scaler = MinMaxScaler()
	num_cols_data = scaler.fit_transform(num_cols)
	num_cols = pd.DataFrame(
		data = num_cols_data,
		columns=num_cols.columns
	)
		
	# One-hot-encoding the categorical features
	if encode_cat:
		encoder = OneHotEncoder(sparse_output=False, drop='first')
		encoded_cats = encoder.fit_transform(cat_cols)
	
		# Converting encoded array to DataFrame
		encoded_df = pd.DataFrame(
			data=encoded_cats,
			columns=encoder.get_feature_names_out(categorical_columns)
		) 
	
	else:
		encoded_df = cat_cols


	# Combining with numerical columns
	final_data = pd.concat([num_cols.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

	y = y.cat.rename_categories(list(range(n)))
	y = y.fillna(y.mode()[0])

	return final_data, y




def encode_columns(data:pd.DataFrame, columns:list)-> pd.DataFrame:
	'''
	
	'''

	# Choosing the selected columns then excludind the numerical ones
	data = data.loc[:, columns]
	cat_cols = data.select_dtypes(include=['object', 'category']).columns
	# Selecting the other columns
	other_cols = data.drop(columns=cat_cols)

	# One-hot-encoding the categorical features
	encoder = OneHotEncoder(sparse_output=False, drop='first')
	encoded_cats = encoder.fit_transform(data[cat_cols])

	# Converting encoded array to DataFrame
	encoded_df = pd.DataFrame(
		data=encoded_cats,
		columns=encoder.get_feature_names_out(cat_cols)
	) 
	

	# Combining with numerical columns
	final_data = pd.concat([other_cols.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

	return final_data, list(final_data.columns)



def euclidian_dist(x:pd.DataFrame, y:pd.DataFrame) -> float:
	'''
		Evaluates the euclidian distance between two samples
		## Parameters:
		- x, y : samples
	'''

	dist = 0
	for i in range(x.shape[0]):
		try:
			dist += (x.iloc[i] - y.iloc[i])**2
		except:
			if x.iloc[i] != y.iloc[i]:
				dist += 1
		
	return dist**0.5




def hit_and_miss(data:pd.DataFrame, y:pd.Series, a:int) -> tuple[int, int]:
	'''
	Evaluates and finds the nearest hit H and the nearest miss M for sample data[a].
	## Parameters:
	- data: Dataframe-like object. The data
	- y:	Vector-like object. The data labels
	- a:	integer. The row index of the sample to evaluate.
	'''
	H, M = None, None
	scores = [0] * data.shape[0]

	for i in range(data.shape[0]):
		if i != a:
			scores[i] = euclidian_dist(data.iloc[i, :], data.iloc[a, :])
		
	# Looking for the nearest hit and nearest miss
	scores[a] = 1e9
	while H == None or M == None:
		t = np.argmin(scores)
		
		if y[t] == y[a]:
			if H == None:
				H = t
		else:
			if M == None:
				M = t

		scores[t] = 1e9
	
	# print('Done searching nearest Hit and Miss')

	return H, M




def k_hits_or_misses(data:pd.DataFrame, y:pd.Series, a:int, k:int, m=None) -> tuple[int, int]:
	'''
	Evaluates and finds the k nearest hits or misses for sample data[a].
	## Parameters:
	- data: Dataframe-like object. The data
	- y:	Vector-like object. The data labels
	- a:	integer. The row index of the sample to evaluate.
	- k:	integer. The number of hits to look for.
	- m:	integer. The class of the misses to look for.
	'''
	H, M = [], []
	scores = [0] * data.shape[0]

	for i in range(data.shape[0]):
		if i != a:
			scores[i] = euclidian_dist(data.iloc[i, :], data.iloc[a, :])
		
	# Looking for the nearest hit and nearest miss
	scores[a] = 1e9
	n = y.shape[0]
	while n > 0:
		t = np.argmin(scores)
		
		if m == None:
			if y[t] == y[a]:
				if len(H) < k:
					H.append(t)
		else:
			if y[t] != y[a] and y[t] == m:
				if len(M) < k:
					M.append(t)
		
		n -= 1
		
		scores[t] = 1e9
	
	# print('Done searching k-nearest Hits/Misses')

	return H if m == None else M




def k_hit_and_miss(data:pd.DataFrame, y:pd.Series, a:int, k:int) -> tuple[int, int]:
	'''
	Evaluates and finds the nearest hit H and the nearest miss M for sample data[a].
	## Parameters:
	- data: Dataframe-like object. The data
	- y:	Vector-like object. The data labels
	- a:	integer. The row index of the sample to evaluate.
	- k:	integer. The number of hits and misses to look for.
	'''
	H, M = [], []
	scores = [0] * data.shape[0]

	for i in range(data.shape[0]):
		if i != a:
			scores[i] = euclidian_dist(data.iloc[i, :], data.iloc[a, :])
		
	# Looking for the nearest hit and nearest miss
	scores[a] = 1e9
	n = y.shape[0]
	while n > 0:
		t = np.argmin(scores)
		
		if y[t] == y[a]:
			if len(H) < k:
				H.append(t)
		else:
			if len(M) < k:
				M.append(t)

		n -= 1

		scores[t] = 1e9
	
	# print('Done searching nearest Hit and Miss')

	return H, M




def diff(data:pd.DataFrame, A:int, I1:int, I2:int) -> float:
	'''
	
	'''
	try: # For numerical attributes A
		return np.abs(data.iloc[I1, A] - data.iloc[I2, A]) / (np.max(data.iloc[:, A]) - np.min[:, A])

	except: # For nominal attributes A
		return int(not data.iloc[I1, A] == data.iloc[I2, A])



def kmeans_discretization_engine(data:pd.Series) -> dict:
	'''
		Discretizes a numerical attribute using K-means clustering and silhouette score to determine the optimal number of bins.
		## Parameters
		- data:  Series-like object. The numerical attribute to discretize.
		## Returns
		- optimal_discretization: dict. A dictionary containing the optimal number of bins,
		  the corresponding KMeans model, and the silhouette score.
	'''
	
	# Convert to 2D
	X = data.reshape(-1, 1)

	optimal_discretization = {
			'score' : -1,
			'engine' : None,
			'n_bins' : 0
	}
	
	for i in  range(2, 10):
		kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
		kmeans.fit(X)
		cluster_labels = kmeans.predict(X)
		
		silhouette_avg = silhouette_score(X, cluster_labels)
		
		if silhouette_avg > optimal_discretization['score']:
			optimal_discretization['score'] = silhouette_avg
			optimal_discretization['engine'] = kmeans
			optimal_discretization['n_bins'] = i
	
	
	return optimal_discretization



def entropy(x: pd.Series) -> float:
	"""
	Shannon entropy H(X).
	
	## Parameters
	x: vector-like object. The variable for which to calculate the entropy.
	## Returns
	float. The calculated entropy.

	"""
	p = pd.Series(x).value_counts(normalize=True)
	return -(p * np.log2(p)).sum()


def conditional_entropy(x: pd.Series, y: pd.Series) -> float:
	"""
	H(X|Y)
	
	## Parameters
	x: vector-like object. The variable for which to calculate the conditional entropy.
	y: vector-like object. The variable to condition on.
	## Returns
	float. The calculated conditional entropy.

	"""
	df = pd.DataFrame({'x': x, 'y': y})
	
	h = 0
	for val, subset in df.groupby('y', observed=True):
		p = len(subset) / len(df)
		h += p * entropy(subset['x'])
		
	return h


def conditional_mutual_information(a: pd.Series, b: pd.Series, c: pd.Series) -> float:
	"""
	I(A;B|C)
	
	## Parameters
	a: vector-like object. The first variable.
	b: vector-like object. The second variable.
	c: vector-like object. The conditioning variable.
	## Returns
	float. The calculated conditional mutual information.
	"""
	h_b_c = conditional_entropy(b, c)
	h_b_ac = conditional_entropy(b, pd.Series(list(zip(a, c))))
	return h_b_c - h_b_ac


def mutual_information(a: pd.Series, b: pd.Series) -> float:
	"""
	Compute mutual information I(A;B) between two discrete variables.
	
	Parameters:
		a (pd.Series): First variable
		b (pd.Series): Second variable
		
	Returns:
		float: Mutual information in bits
	"""
	
	# Drop missing values jointly
	df = pd.DataFrame({'a': a, 'b': b}).dropna()
	
	# Joint distribution
	joint = pd.crosstab(df['a'], df['b'])
	joint = joint / joint.values.sum()
	
	# Marginals
	p_a = joint.sum(axis=1).values.reshape(-1, 1)
	p_b = joint.sum(axis=0).values.reshape(1, -1)
	
	expected = p_a @ p_b
	
	joint_vals = joint.values
	mask = joint_vals > 0
	
	return np.sum(joint_vals[mask] * np.log2(joint_vals[mask] / expected[mask]))


def theils_u(a: pd.Series, b: pd.Series) -> float:
	"""
	U(A , B)
	
	## Parameters
	a: vector-like object. The first variable.
	b: vector-like object. The second variable.
	## Returns
	float. The calculated conditional Theil's U.
	"""
	h_b = entropy(b)
	
	if h_b == 0:
		return 0
	
	mi = mutual_information(a, b)
	
	return mi / h_b




def build_graph(data:pd.DataFrame, class_labels:pd.Series, weights_strategy:str= 'corcoef', gamma:float=1.0) -> nx.DiGraph:
	'''
		Build a complete directed weighted graph from the data. The nodes represent the features and the weights
		a similtude between the nodes.
		## Parameters:
		- data	: matrix-like object. The data from which the graph is built.
		- class_labels: vector-like object. The class labels of the data.
		- weights_strategy:	string. The weighting stategy (corcoef, mi, theilsU). Defaults to corcoef.
		- gamma: float. The calibration factor for the asymmetrical weight calculation. Defaults to 0.85.

		## Returns:
		- graph: NetworkX graph object. The built graph.
	
	'''

	# print(f"Building features' graph using {weights_strategy} strategy.")
	n = data.shape[1]
	graph_matrix = np.zeros((n, n), 'float64')
	numeric_data = data.select_dtypes(exclude=['object', 'category'])
	categorical_data = data.select_dtypes(include=['object', 'category'])


	# Discretising the numerical features using a decision tree discretiser
	disc = DecisionTreeDiscretiser(
		cv=3,
		scoring='neg_mean_squared_error',
		variables=numeric_data.columns.tolist(),
		param_grid={'max_depth': [1, 2, 3]},
		bin_output='bin_number',
		regression=False
	)

	# Fitting the discretiser to the data and transforming it
	disc.fit(numeric_data, class_labels)

	# Transforming the numerical data into binned data
	numeric_binned = disc.transform(numeric_data)

	# Combining the binned numerical data with the categorical data
	data_cat = pd.concat([numeric_binned, categorical_data], axis=1)


	for i in range(n):
		for j in range(n):

			if i != j:

				if weights_strategy == 'corcoef': # Correlation coefficient (absolute value is used to represent the dependence without orientation)
					
					# Symetrical weight calculation
					if i < j:
						graph_matrix[i, j] = graph_matrix[j, i] = np.abs(data[[data.columns[i], data.columns[j]]].corr('pearson').to_numpy()[0, 1])
					
					# Asymetrical weight calculation
					else:
						graph_matrix[i, j] = (1 - gamma) * graph_matrix[i, j] + gamma * conditional_mutual_information(data_cat[data_cat.columns[j]], data_cat[data_cat.columns[i]], class_labels)



				elif weights_strategy == 'mi': # Mutual information
					
					# Symetrical weight calculation
					if i < j:
						tmp = mutual_info_regression(data_cat[[data_cat.columns[i]]], data_cat[data_cat.columns[j]])

						# Normalising the values
						div = (mutual_info_regression(data_cat[[data_cat.columns[i]]], data_cat[data_cat.columns[i]]) * mutual_info_regression(data_cat[[data_cat.columns[j]]], data_cat[data_cat.columns[j]]))
						if div == 0:
							graph_matrix[i, j] = graph_matrix[j, i] = tmp
						else:
							graph_matrix[i, j] = graph_matrix[j, i] = tmp / div ** 0.5

					# Asymetrical weight calculation
					else:
						graph_matrix[i, j] = (1 - gamma) * graph_matrix[i, j] + gamma * conditional_mutual_information(data_cat[data_cat.columns[j]], data_cat[data_cat.columns[i]], class_labels)

						# Normalising the values
						graph_matrix[i, j] /= 2


				else: # Theil's U
					
					# Asymetrical weight calculation independant of the class labels
					if i < j:
						tmp = theils_u(data_cat[data_cat.columns[i]], data_cat[data_cat.columns[j]])

						# Normalising the values
						# graph_matrix[i, j] = graph_matrix[j, i] = tmp / (theils_u(data_cat[data_cat.columns[i]], data_cat[data_cat.columns[i]]) * theils_u(data_cat[data_cat.columns[j]], data_cat[data_cat.columns[j]])) ** 0.5

					# Asymetrical weight calculation
					else:
						graph_matrix[i, j] = (1 - gamma) * graph_matrix[i, j] + gamma * conditional_mutual_information(data_cat[data_cat.columns[j]], data_cat[data_cat.columns[i]], class_labels)

						# Normalising the values
						graph_matrix[i, j] /= 2



	graph_matrix = np.clip(graph_matrix, 0.0, 1.0)
	return nx.from_numpy_array(graph_matrix, parallel_edges=True, create_using=nx.DiGraph)

