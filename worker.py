'''
worker.py

This script manages datasets and their associated metadata.

By Alph@B, AKA Brel MBE & Arielle Kana
'''

import sys
import getopt
import os




DATASETS = {
	0: "Breast cancer",
	1: "Contraceptive method choice",
	2: "Credit risk data",
	3: "Glass Identification",
	4: "Speaker Accent Recognition",
	5: "Statlog Australian Credit Approval",
	6: "German Credit",
	7: "Ionosphere",
	8: "Mini Credit Risk",
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
	8: "PageRank with weightdrop1 strategy",
	9: "PageRank with weightdrop2 strategy",
	10: "PageRank with deletion strategy",
	11: "UGFS",
	12: "PPRFS",
	13: "MGFS",
	14: "SGFS",
	15: "FSS-CPR"
	
}

CLASSIFIERS = {
	0: 'SVM',
	1: 'LogReg',
	2: 'DecTree',
	3: 'RanForests',
	4: 'HistGradBoost',
	5: 'LinDiscrimAnalysis',
}

GAMMA_VALUES = [0.0, 0.2, 0.5, 0.85, 1.0]
N_FEATURES_RANGE = 10

try:
	cpts, args = getopt.getopt(sys.argv[1:], "d:", ["dataset="])

except getopt.GetoptError as err:
	print(err)
	sys.exit(2)



# algo = -1
classifier = 0
dataset = 0

for o, a in cpts:
	
	if o in ('-d', '--dataset'):
		try:
			dataset = int(a)
		except ValueError:
			print('dataset parameter must be an integer.')
			sys.exit(2)
	else:
		print(f"Option {o} inconnue.")
		sys.exit(2)


try:
	os.mkdir(f"reports/RAW_TXT")
except:
	pass


for classifier in CLASSIFIERS.keys():

	for gamma in GAMMA_VALUES:

		file_path = f"reports/RAW_TXT/{CLASSIFIERS[classifier]}/dataset_{dataset}_classifier_{classifier}_gamma={str(gamma)}.txt"
		
		try:
			os.mkdir(f"reports/RAW_TXT/{CLASSIFIERS[classifier]}")
		except:
			pass

		open(file_path, "w+")
		

		with open(file_path, "w+") as file:
			file.write(f"#################### Dataset : {DATASETS[dataset]} - Classifier : {CLASSIFIERS[classifier]} ####################\n\n")

		for algo in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:


			if dataset in [1, 3, 4] and algo == 0: # Multiclass dataset with relief
				pass

			elif dataset not in [1, 3, 4] and algo == 1: # Biclass dataset with reliefF
				pass

			else:

				print(f"Feature selection algo : {ALGOS[algo]}")

				# ------------------------------------------------------------------
				# RELIEF / RELIEFF
				# ------------------------------------------------------------------
				if algo in [0, 1]:

					with open(file_path, "a+") as file:
						file.write(f"####################################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
					
					for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
						
						print(f"Number of features = {int(n_features * 100)}%")

						with open(file_path, "a+") as file:
							file.write(f"\nNumber of features = {int(n_features * 100)}%")
						
						for m in [5, 20, 50, 90]:
							os.system(f"python main.py -d {dataset} -a {algo} -c {str(classifier)} -p {m} -n {n_features} -g {gamma}")

				# ------------------------------------------------------------------
				# RIDGE / LASSO
				# ------------------------------------------------------------------
				elif algo in [6, 7]:

					with open(file_path, "a+") as file:
						file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
					
					for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
						
						print(f"Number of features = {int(n_features * 100)}%")

						with open(file_path, "a+") as file:
							file.write(f"\nNumber of features = {int(n_features * 100)}%")
						
						for m in [0.00001, 0.01, 0.1, 1, 5, 20, 50]:
							os.system(f"python main.py -d {dataset} -a {algo} -c {str(classifier)} -p {m} -n {n_features} -g {gamma}")

				# ------------------------------------------------------------------
				# PAGERANK ORIGINAL (algos 8, 9 and 10 are variants of pagerank)
				# ------------------------------------------------------------------
				elif algo in [8, 9, 10]: # PageRank with weightdrop1, weightdrop2 and deletion strategies

					with open(file_path, "a+") as file:
						file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
					
					for weighing_strat in ['corcoef', 'mi', 'theils_u']:
						
						with open(file_path, "a+") as file:
							file.write(f"\n#################### Graph weighting strategy: {weighing_strat} ####################\n\n")

						print(f"Graph weighting strategy: {weighing_strat}")

						for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
							
							print(f"Number of features = {int(n_features * 100)}%")

							with open(file_path, "a+") as file:
								file.write(f"\nNumber of features = {int(n_features * 100)}%")
							
							# for m in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
							for m in [0.05, 0.3, 0.6, 0.95]:
								os.system(f"python main.py -d {dataset} -a {algo} -c {str(classifier)} -p {m} -n {n_features} -s {weighing_strat} -g {gamma}")
				
				# ------------------------------------------------------------------
				# NOUVEAUX ALGORITHMES DES ARTICLES (11–15)
				# ------------------------------------------------------------------
				# elif algo in [11, 12, 13, 14, 15]:

				# 	# Ici on ne passe PAS de stratégie de graphe externe
				# 	# car chaque algorithme construit son propre graphe
				# 	# conformément aux articles scientifiques.

				# 	with open(file_path, "a+") as file:
				# 		file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")

				# 	for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:

				# 		print(f"Number of features = {int(n_features * 100)}%")

				# 		with open(file_path, "a+") as file:
				# 			file.write(f"\nNumber of features = {int(n_features * 100)}%")

				# 		os.system(f"python main.py -d {dataset} -a {algo} -c {str(classifier)} -n {n_features} -g {gamma}")
				
				# ------------------------------------------------------------------
				# AUTRES ALGORITHMES (NOUVEAUX OU ANCIENS)
				# ------------------------------------------------------------------
				else:

					with open(file_path, "a+") as file:
						file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
					
					for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:

						print(f"Number of features = {int(n_features * 100)}%")

						with open(file_path, "a+") as file:
							file.write(f"\nNumber of features = {int(n_features * 100)}%")

						os.system(f"python main.py -d {dataset} -a {algo} -c {str(classifier)} -n {n_features} -g {gamma}")

						
