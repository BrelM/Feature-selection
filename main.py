'''
	main.py

	Lead the parameted execution of all FS techniques.


	By Alph@B, AKA Brel MBE

'''

import math
import sys
import getopt
import networkx as nx

# import numpy as np

import utils
import algorithms
import classifiers
import pagerank


DATA_PATH = "../Data"

DATASETS = {
	0: "Breast cancer",
	1: "Contraceptive method choice",
	2: "Credit risk data",
	3: "Glass Identification",
	4: "Speaker Accent Recognition",
	5: "Statlog Australian Credit Approval",
	6: "German Credit",
	7: "Ionosphere",
}

ALGOS = {
	0: "Relief",
	1: "ReliefF",
	2: "Mutual information",
	3: "Sequential feature selection",
	4: "RFE-SVM",
	5: "RFE-SVM-SFS",
	6: "RIDGE",
	7: "LASSO",
	8: "PageRank",
	9: "PageRank del",
}

ALGOS_INFO = {
	0: algorithms.relief,
	1: algorithms.reliefF,
	2: algorithms.mutual_info,
	3: algorithms.forward_FS,
	4: algorithms.svm_rfe	,
	5: algorithms.svm_rfe_sfs,
	6: algorithms.ridge_fs,
	7: algorithms.lasso_fs,
	8: pagerank.pagerankloop,
	9: pagerank.pagerankloop,
}




CLASSIFIERS = {
	0: 'SVM',
	1: 'Logistic regression',
	2: 'Decision tree',
	3: 'Random forests',
	4: 'Histogram gradient boosting',
	5: 'Linear Discriminative analysis',

}


CLASSIFIERS_INFO = {
	0: classifiers.svm_classifier,
	1: classifiers.logreg_classifier,
	2: classifiers.dectree_classifier,
	3: classifiers.randforest_classifier,
	4: classifiers.higradboost_classifier,
	5: classifiers.lda_classifier,

}




DATASETS_INFO = {
	0: {
			'name': "Breast cancer",
			'path': DATA_PATH + '/' + "breast+cancer+wisconsin+diagnostic/wdbc.data",
			'nb_features': 32,
			'class_idx': 1,
			'sep': ',',
			'id': True,
			"labels": False,
			"categorical": False,
		},
	1: {
			'name': "Contraceptive method choice",
			'path': DATA_PATH + '/' + "contraceptive+method+choice/cmc.data",
			'nb_features': 10,
			'class_idx': 9,
			'sep': ',',
			'id': False,
			"labels": False,
			"categorical": False,
		},
	2: {
			'name': "Credit risk data",
			'path': DATA_PATH + '/' + "Credit risk data/credit_risk_dataset.csv",
			'nb_features': 12,
			'class_idx': 10,
			'sep': ',',
			'id': False,
			"labels": True,
			"categorical": True,
		},
	3: {
			'name': "Glass Identification",
			'path': DATA_PATH + '/' + "glass+identification/glass.data",
			'nb_features': 11,
			'class_idx': 10,
			'sep': ',',
			'id': True,
			"labels": False,
			"categorical": False,
		},

	4: {
			'name': "Speaker Accent Recognition",
			'path': DATA_PATH + '/' + "speaker+accent+recognition/accent-mfcc-data-1.csv",
			'nb_features': 13,
			'class_idx': 0,
			'sep': ',',
			'id': False,
			"labels": True,
			"categorical": False,
		},
	5: {
			'name': "Statlog Australian Credit Approval",
			'path': DATA_PATH + '/' + "statlog+australian+credit+approval/australian.dat",
			'nb_features': 15,
			'class_idx': 14,
			'sep': ' ',
			'id': False,
			"labels": False,
			"categorical": False,
		},
	6: {
			'name': "German Credit",
			'path': DATA_PATH + '/' + "statlog+german+credit+data/german.data",
			'nb_features': 21,
			'class_idx': 20,
			'sep': ' ',
			'id': False,
			"labels": False,
			"categorical": True,
		},

	7: {
			'name': "Ionosphere",
			'path': DATA_PATH + '/' + "ionosphere/ionosphere.data",
			'nb_features': 35,
			'class_idx': 34,
			'sep': ',',
			'id': False,
			"labels": False,
			"categorical": False,
		},

}




def usage():
	print("Execute a feature selection algorithm over a specified dataset.\n\
Store the resuts in a some files.\n\
For more support:\n\
-h --help   :\t get help\n\
-a --algo   :\t specify the algorithm to use\n\
-d --dataset:\t specify the dataset to use\n\
-n --n_features:\t the percentage of features to select \n\
-p --params:\t specify the feature selection parameters to use\n\
-c --classif:\t specify the classifier to use for evaluation of the features (before and after selection) \n\
-s			:\t For PageRank-based feature selection algorithm. The graph weighting strategy to use: corcoef, mi\n\
\n\
Feature selection algorihms include:\n\
{}\n\
Datasets include:\n\
{}\n\
Classifiers include:\n\
{}\n\
".format(ALGOS, DATASETS, CLASSIFIERS))




try:
	cpts, args = getopt.getopt(sys.argv[1:], "ha:d:p:c:n:s:", ["help", "algo=", "dataset=", "params=", "classif=", "n_features="])

except getopt.GetoptError as err:
	print(err)
	usage()
	sys.exit(2)



algo = -1
dataset = -1
classifier = -1
params = None
strategy = 'corcoef'
n_features = 0.5

for o, a in cpts:
	
	if o == '-h':
		usage()
		sys.exit(0)

	elif o in ('-a', '--algo'):
		try:
			algo = int(a)
			if algo not in ALGOS_INFO.keys():
				print(f"algo parameter must be an integer between 0 and {len(ALGOS_INFO.keys())}.")
				sys.exit(2)
		except ValueError:
			print('algo parameter must be an integer.')
			sys.exit(2)

	elif o in ('-d', '--dataset'):
		try:
			dataset = int(a)
			if dataset not in DATASETS_INFO.keys():
				print(f"dataset parameter must be an integer between 0 and {len(DATASETS_INFO.keys())}.")
				sys.exit(2)
		except ValueError:
			print('dataset parameter must be an integer.')
			sys.exit(2)
	
	elif o in ('-p' '--params'):
		params = a
	
	elif o == '-s':
		strategy = a
	
	elif o in ('-n', '--n_features'):
		n_features = float(a)

	elif o in ('-c' '--classif'):
		try:	
			classifier = int(a)
			if classifier not in CLASSIFIERS_INFO.keys():
				print(f"classif parameter must be an integer between 0 and {len(CLASSIFIERS_INFO.keys())}.")
				sys.exit(2)
		except ValueError:
			print('classif parameter must be an integer.')
			sys.exit(2)

	else:
		print(f"Option {o} inconnue.")
		sys.exit(2)

if dataset == -1 or algo == -1 or classifier == -1:
	usage()
	sys.exit(2)


data, y = utils.load_data(DATASETS_INFO[dataset])
total_nb_feat = DATASETS_INFO[dataset]['nb_features']


if params == None and algo in [6, 7]: # Ridge or Lasso
	params = 1e-5

if params == None and algo in [8, 9]: # PageRank
	params = 0.85

if params != None:

	# if '.' in params:
	try:
		params = int(params)
	except:# TypeError as e:
		params = float(params)



if math.ceil(n_features * total_nb_feat) == total_nb_feat:
	n_features = math.ceil(n_features * total_nb_feat) - 1
else:
	n_features = math.ceil(n_features * total_nb_feat)


# Some infos about the data
# print(data.info(), '\n', y.cat.categories)



Data, Y = None, None

if n_features != total_nb_feat - 1: # Feature selection to apply

	# Execute the choosen algorithm
	if algo in [8, 9]:
		Data, Y = utils.load_data(DATASETS_INFO[dataset], False)
		graph = utils.build_graph(Data, strategy)

		if algo == 8:
			columns = ALGOS_INFO[algo](graph, list(Data.columns), alpha=params, max_iter=n_features, pen_method="penalize")
		else:
			columns = ALGOS_INFO[algo](graph, list(Data.columns), alpha=params, max_iter=n_features, pen_method="delete")



	elif algo in [0, 1, 6, 7]:
		columns = ALGOS_INFO[algo](data, y, params, n_features=n_features)
	
	else:
		columns = ALGOS_INFO[algo](data, y, n_features=n_features)
	

	#######################################################################
	## In case of encoded features...
	if DATASETS_INFO[dataset]['categorical']:

		## Cleaning isolated features' derivatives
		display_cols_, temp = [], list(columns)
		for c in temp:
			cooked = False
			idx = c.find('_')
			if idx >= 0:
				
				for _ in list(data.columns):
				
					if c[:idx] + '_' in _:
						if _ not in temp:
							cooked = True
							break
				if cooked:
					columns.remove(c)
				else:
					display_cols_.append(c[:idx])
			else:
				display_cols_.append(c) 

					
	
		## Selecting all of a feature's derivatives if at least one is selected
		full_columns, display_cols = [], []
		for c in temp:

			idx = c.find('_')
			if idx >= 0:
				display_cols.append(c[:idx])
				for _ in list(data.columns):
					
					if c[:idx] + '_' in _:
						if _ not in full_columns:
							full_columns.append(_)
				
			else:
				display_cols.append(c)
				full_columns.append(c)




else: # No feature selection to apply

	if algo in [8, 9]:
		Data, Y = utils.load_data(DATASETS_INFO[dataset], False)
		columns = list(Data.columns)
	else:
		columns = list(data.columns)
		temp_cols = []
		for col in columns:
			idx = col.find("_")
			if idx >= 0:
				if col[:idx] not in temp_cols:
					temp_cols.append(col[:idx])
			else:
				temp_cols.append(col)
		columns = list(temp_cols)


with open(f"reports/dataset_{dataset}.txt", "a+") as file:

	file.write('\n')
	# print(f"Feature selection algorithm: {ALGOS[algo]}\n
	file.write(f"\nMeta parameter(s) value(s): {params}\n\n")

	if DATASETS_INFO[dataset]['categorical']: # Two sets of selected features

		if n_features != total_nb_feat - 1: # Feature selection to apply
			for message, process_cols, disp_cols in [("Soft selection of feature derivatives", full_columns, display_cols), ("Strict selection of feature derivatives", columns, display_cols_)]:

				file.write(message + '\n')
				if process_cols == []: # Case isolated derivatives where selected and ruled out (strict selection)
					file.write(f"Selected features: None (Isolated derivatives {temp})\n")
					file.write(f"Accuracy, f1-score: 0, 0\n")

				else:
					file.write(f"Selected features: {disp_cols}\n")
					if algo in [8, 9]:

						encoded_data, columns = utils.encode_columns(Data, process_cols)
						file.write(f"Accuracy, f1-score: {CLASSIFIERS_INFO[classifier](encoded_data, y)}\n")
					
					else:
						file.write(f"Accuracy, f1-score: {CLASSIFIERS_INFO[classifier](data, y, process_cols)}\n")
		else:
			file.write(f"Selected features: {columns}\n")
			if algo in [8, 9]:

				file.write(f"Accuracy, f1-score: {CLASSIFIERS_INFO[classifier](data, y)}\n")
			
			else:
				file.write(f"Accuracy, f1-score: {CLASSIFIERS_INFO[classifier](data, y)}\n")

	else:

		file.write(f"Selected features: {columns}\n")
		if algo in [8, 9]:

			encoded_data, columns = utils.encode_columns(Data, columns)
			file.write(f"Accuracy, f1-score: {CLASSIFIERS_INFO[classifier](encoded_data, y)}\n\n")
		
		else:
			file.write(f"Accuracy, f1-score: {CLASSIFIERS_INFO[classifier](data, y, columns)}\n\n")






