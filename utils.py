'''
	utils.py

	Some useful functions and procedures.


	By Alph@B, AKA Brel MBE

'''



import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



def load_data(data_params:dict) -> tuple[pd.DataFrame, pd.Series]:
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
		num_cols[col] = data[col].fillna(mean_value)


	cat_cols = data[categorical_columns]
	for col in categorical_columns:
		mode_value = data[col].mode()[0]
		cat_cols[col] = data[col].fillna(mode_value)


	# Scaling numerical values
	scaler = MinMaxScaler()
	num_cols_data = scaler.fit_transform(num_cols)
	num_cols = pd.DataFrame(
		data = num_cols_data,
		columns=num_cols.columns
	)
		
	# One-hot-encoding the categorical features
	encoder = OneHotEncoder(sparse_output=False, drop='first')
	encoded_cats = encoder.fit_transform(cat_cols)

	# Converting encoded array to DataFrame
	encoded_df = pd.DataFrame(
		data=encoded_cats,
		columns=encoder.get_feature_names_out(categorical_columns)
	) 

	# Combining with numerical columns
	final_data = pd.concat([num_cols.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

	return final_data, y.cat.rename_categories(list(range(n)))




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
	
	print('Done searching nearest Hit and Miss')

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
	
	print('Done searching nearest Hit and Miss')

	return H, M




def diff(data:pd.DataFrame, A:int, I1:int, I2:int) -> float:
	'''
	
	'''
	try: # For numerical attributes A
		return np.abs(data.iloc[I1, A] - data.iloc[I2, A]) / (np.max(data.iloc[:, A]) - np.min[:, A])

	except: # For nominal attributes A
		return int(not data.iloc[I1, A] == data.iloc[I2, A])



def build_graph(data:pd.DataFrame, weights_strategy:str= 'corcoef') -> np.array:# | ['corcoef', 'mi', 'chi2']):
	'''
		Build a complete weighted graph rom the data. The nodes represent the features and the weights
		a similtude between the nodes.
		## Parameters:
		- Data	: matrix-like object. The data from which the graph is built.
		- weights_strategy:	string. The weighting stategy (corcoef, mi, chi2). Defaults to corcoef.
	
	'''

	n = data.shape[1]
	graph_matrix = np.zeros([n, n], 'float64')

	if weights_strategy == 'corcoef':
		return data.corr('pearson').to_numpy()

	for i in range(n):
		for j in range(n):
			graph_matrix[i, j] = mutual_info_regression(data[[data.columns[i]]], data[data.columns[j]])


	return nx.from_numpy_array(graph_matrix, parallel_edges=False)