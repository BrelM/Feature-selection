

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
	9: "PageRank with deletion strategy",
}


N_FEATURES_RANGE = 10

try:
	cpts, args = getopt.getopt(sys.argv[1:], "d:", ["dataset="])

except getopt.GetoptError as err:
	print(err)
	sys.exit(2)



algo = -1
dataset = -1

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



with open(f"reports/dataset_{dataset}.txt", "w+") as file:
	file.write(f"#################### Dataset : {DATASETS[dataset]} ####################\n\n")



for algo in [5, 6, 7, 8, 9]:#[0, 1, 2, 3, 5, 6, 7, 8, 9]:

	if dataset in [1, 3, 4] and algo == 0: # Multiclass dataset with relief
		pass

	elif dataset not in [1, 3, 4] and algo == 1: # Biclass dataset with reliefF
		pass

	else:

		print(f"Feature selection algo : {ALGOS[algo]}")
		if algo in [0, 1]: # relief or reliefF

			with open(f"reports/dataset_{dataset}.txt", "a+") as file:
				file.write(f"####################################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
			
			for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
				
				print(f"Number of features = {int(n_features * 100)}%")
				with open(f"reports/dataset_{dataset}.txt", "a+") as file:
					file.write(f"\nNumber of features = {int(n_features * 100)}%")
				
				for m in [5, 20, 50, 90]:
					os.system(f"python main.py -d {dataset} -a {algo} -c 0 -p {m} -n {n_features}")
		
		
		elif algo in [6, 7]: # ridge or lasso

			with open(f"reports/dataset_{dataset}.txt", "a+") as file:
				file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
			
			for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
				
				print(f"Number of features = {int(n_features * 100)}%")
				with open(f"reports/dataset_{dataset}.txt", "a+") as file:
					file.write(f"\nNumber of features = {int(n_features * 100)}%")
				
				for m in [0.00001, 0.01, 0.1, 1, 5, 20, 50]:
					os.system(f"python main.py -d {dataset} -a {algo} -c 0 -p {m} -n {n_features}")

		elif algo in [8, 9]: # PageRank

			with open(f"reports/dataset_{dataset}.txt", "a+") as file:
				file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
			
			for weighing_strat in ['corcoef', 'mi']:
				
				with open(f"reports/dataset_{dataset}.txt", "a+") as file:
					file.write(f"\n#################### Graph weighting strategy: {weighing_strat} ####################\n\n")
				print(f"Graph weighting strategy: {weighing_strat}")

				for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
					
					print(f"Number of features = {int(n_features * 100)}%")
					with open(f"reports/dataset_{dataset}.txt", "a+") as file:
						file.write(f"\nNumber of features = {int(n_features * 100)}%")
					
					for m in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
						os.system(f"python main.py -d {dataset} -a {algo} -c 0 -p {m} -n {n_features} -s {weighing_strat}")


		else:

			with open(f"reports/dataset_{dataset}.txt", "a+") as file:
				file.write(f"###########################################################################\n#################### Feature selection algo : {ALGOS[algo]} ####################\n")
			
			for n_features in [(i + 1)/10 for i in range(N_FEATURES_RANGE)]:
				
				print(f"Number of features = {int(n_features * 100)}%")
				with open(f"reports/dataset_{dataset}.txt", "a+") as file:
					file.write(f"\nNumber of features = {int(n_features * 100)}%")
				os.system(f"python main.py -d {dataset} -a {algo} -c 0 -n {n_features}")








