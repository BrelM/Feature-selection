'''
	plotting.py

	For plotting built graphs.


	By Alph@B, AKA Brel MBE

'''



import utils
import sys
import getopt
import matplotlib.pyplot as plt
import networkx as nx



DATA_PATH = "../Data"
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






dataset = -1
strategy = 'mi'

try:
	cpts, args = getopt.getopt(sys.argv[1:], "d:", ["dataset="])

except getopt.GetoptError as err:
	print(err)
	sys.exit(2)

for o, a in cpts:
	
	if o in ('-d', '--dataset'):
		try:
			dataset = int(a)
		except ValueError:
			print('Dataset parameter must be an integer.')
			sys.exit(2)
	else:
		print(f"Option {o} inconnue.")
		sys.exit(2)




# Loading the choosen dataset
print(f"Loading the dataset: {DATASETS_INFO[dataset]['name']}") 
data, y = utils.load_data(DATASETS_INFO[dataset])
total_nb_feat = DATASETS_INFO[dataset]['nb_features']


# Building a graph from the dataset
print("Buidling the graph...")
Data, Y = utils.load_data(DATASETS_INFO[dataset], False)
G = utils.build_graph(Data, strategy)



# Drawing the graph
print("Plotting the graph...")
pos = nx.circular_layout(G)
weights = nx.get_edge_attributes(G, 'weight')

# plt.figure(1)
plt.figure(1, figsize=(15, 8))

nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8)

plt.savefig(f"Graph_{DATASETS_INFO[dataset]['name'].replace(' ', '_')}_{strategy}.pdf")
plt.show()




