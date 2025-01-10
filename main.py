'''
	main.py

	Lead the parameted execution of all FS techniques.


	By Alph@B, AKA Brel MBE

'''

import os
import sys
import getopt

# import numpy as np

import utils
import algorithms
import classifiers

DATA_PATH = "../Data"

ALGOS = {
	0: "Relief",
	1: "ReliefF",
	2: "Mutual information",
	3: "Sequential feature selection",
	4: "RFE-SVM",
	5: "RIDGE",
	6: "LASSO"
}

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

ALGOS_INFO = {
	0: algorithms.relief,
	1: algorithms.reliefF,
	2: algorithms.mutual_info,
	3: algorithms.forward_FS,
	4: algorithms.svm_rfe	,
	5: algorithms.svm_rfe_sfs,
	6: algorithms.ridge_fs,
	7: algorithms.lasso_fs
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
			"labels": False
		},
	1: {
			'name': "Contraceptive method choice",
			'path': DATA_PATH + '/' + "contraceptive+method+choice/cmc.data",
			'nb_features': 10,
			'class_idx': 9,
			'sep': ',',
			'id': False,
			"labels": False
		},
	2: {
			'name': "Credit risk data",
			'path': DATA_PATH + '/' + "Credit risk data/credit_risk_dataset.csv",
			'nb_features': 12,
			'class_idx': 10,
			'sep': ',',
			'id': False,
			"labels": True
		},
	3: {
			'name': "Glass Identification",
			'path': DATA_PATH + '/' + "glass+identification/glass.data",
			'nb_features': 11,
			'class_idx': 10,
			'sep': ',',
			'id': True,
			"labels": False
		},

	4: {
			'name': "Speaker Accent Recognition",
			'path': DATA_PATH + '/' + "speaker+accent+recognition/accent-mfcc-data-1.csv",
			'nb_features': 13,
			'class_idx': 0,
			'sep': ',',
			'id': False,
			"labels": True
		},
	5: {
			'name': "Statlog Australian Credit Approval",
			'path': DATA_PATH + '/' + "statlog+australian+credit+approval/australian.dat",
			'nb_features': 15,
			'class_idx': 14,
			'sep': ' ',
			'id': False,
			"labels": False
		},
	6: {
			'name': "German Credit",
			'path': DATA_PATH + '/' + "statlog+german+credit+data/german.data-numeric",
			'nb_features': 24,
			'class_idx': 23,
			'sep': '   ',
			'id': False,
			"labels": False
		},

	7: {
			'name': "Ionosphere",
			'path': DATA_PATH + '/' + "ionosphere/ionosphere.data",
			'nb_features': 35,
			'class_idx': 34,
			'sep': ',',
			'id': False,
			"labels": False
		},

}




def usage():
	print("Execute a feature selection algorithm over a specified dataset.\n\
Store the resuts in a csv file.\n\
For more support:\n\
-h --help   :\t get help\n\
-a --algo   :\t specify the algorithm to use\n\
-d --dataset:\t specify the dataset to use\n\
-p --params:\t specify the feature selection parameters to use\n\
-c --classif:\t specify the classifier to use for evaluation of the features (before and after selection) \n\
\n\
Feature selection algorihms include:\n\
{}\n\
Datasets include:\n\
{}\n\
Classifiers include:\n\
{}\n\
".format(ALGOS, DATASETS, CLASSIFIERS))




try:
	cpts, args = getopt.getopt(sys.argv[1:], "ha:d:p:c:", ["help", "algo=", "dataset=", "params=", "classif="])

except getopt.GetoptError as err:
	print(err)
	usage()
	sys.exit(2)



algo = -1
dataset = -1
classifier = -1
params = None

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


data, y = utils.load_data(DATASETS_INFO[dataset])


if params == None and algo > 5: # Ridge or Lasso
	params = 1e-10

if params != None:
	if '.' in params:
		params = float(params)
	else:
		params = int(params)

# Some infos about the data
# print(data.info(), '\n', y.cat.categories)

# Execute the choosen algorithm
columns = ALGOS_INFO[algo](data, y, params)


print('\n\n\n')
print(f"Feature selection algorithm: {ALGOS[algo]}\nMeta parameter(s) value(s): {params}")

print(f"Selected features: {columns}")
print(f"Classifier for evaluation: {CLASSIFIERS[classifier]}")

print(f"Accuracy before feature selection: {CLASSIFIERS_INFO[classifier](data, y):.2f}")
print(f"Accuracy after feature selection: {CLASSIFIERS_INFO[classifier](data, y, columns):.2f}")








# print(f"No. of features before feature selection: {data.shape[1]}\nNo. of features after fs: {x_train.shape[1]}")
