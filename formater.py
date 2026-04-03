'''
formater.py

This script contains functions to extract and format data from text files.

By Alph@B, AKA Brel MBE
'''

import os
import regex

NUMBER_OF_PGRK_ALPHA_VALUES = 3
NO_PARAMS_ALGOS = [
    "mutual information",
    "sequential feature selection",
    "rfe-svm",
    "rfe-svm-sfs",
    "ugfs",
	"pprfs",
	"mgfs",
	"sgfs",
	"fss-cpr"
]

CLASSIFIERS_FOLDERS = [
    'SVM',
	'LogReg',
	'DecTree',
	'RanForests',
	'HistGradBoost',
	'LinDiscrimAnalysis',
]
TXT_FOLDER = "reports/RAW_TXT"
BASE_FOLDER = "reports/CSV"

def extract_data(file_path, classifier, nb_dataset, gamma):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    folder_path = f"./{BASE_FOLDER}/{classifier}/dataset={nb_dataset}/gamma={gamma}"

    dataset_name = lines[0].strip().split(': ')[1]
    results = []
    current_algo = None
    current_percentage = None
    current_meta_param = None
    current_features = []
    current_accuracy = None
    current_f1_score = None

    a = 0

    eof = False
    while a < len(lines):
        # print(f"{lines[a].casefold()} {a}")

        if "selection algo" in lines[a]:
            current_algo = lines[a].split(':')[1].replace('#', '').strip().casefold()

        if '%' in lines[a]:
            current_percentage = lines[a].split('=')[1].replace('%', '').strip().casefold()
            a += 1 # Go to next line (there's a percentage on the current line)

            if current_algo in NO_PARAMS_ALGOS:
                while 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "feature selection" not in lines[a].casefold():
                    a += 1
            
                if 'accuracy' in lines[a].casefold():
                    current_accuracy = lines[a].split(':')[1].split(', ')[0]
                    current_f1_score = lines[a].split(':')[1].split(', ')[1][:-1]

                    # Making sure the script doesn't crash if the accuracy or f1 score is not found in the text file (in case of an error during the execution of the algorithm for example).
                    current_accuracy = current_accuracy if current_accuracy is not None else "0"
                    current_f1_score = current_f1_score if current_f1_score is not None else "0"
                    
                    m1 = ((current_accuracy + ",") * NUMBER_OF_PGRK_ALPHA_VALUES)[:-1] + "\n"
                    m2 = ((current_f1_score + ",") * NUMBER_OF_PGRK_ALPHA_VALUES)[:-1] + "\n"

                    m1 = f"{current_algo}," + m1
                    m2 = f"{current_algo}," + m2

                    with open(f"{folder_path}/{nb_dataset}_{current_percentage}_accuracy.csv", "a+") as f1:
                        f1.write(m1)
                    with open(f"{folder_path}/{nb_dataset}_{current_percentage}_f1score.csv", "a+") as f1:
                        f1.write(m2)



            if current_algo in ['relief', 'relieff', 'ridge', 'lasso']:

                metrics_sum, metrics = 0, []
                
                while '%' not in lines[a]:
                    if a >= len(lines):
                        eof = True
                        break
                    
                    while 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "feature selection" not in lines[a].casefold():
                        a += 1

                        if a >= len(lines):
                            eof = True
                            break


                    if not eof and 'accuracy' in lines[a].casefold():
                        metrics = [lines[a].split(':')[1].split(', ')[0], lines[a].split(':')[1].split(', ')[1][:-1]]

                        temp = float(metrics[0]) + float(metrics[1])
                        if temp > metrics_sum:
                            metrics_sum = temp
                            current_accuracy = metrics[0]
                            current_f1_score = metrics[1]

                    else:
                        break

                    a += 1

                # print(f"{nb_dataset} - {file_path} - {current_algo} - {current_percentage}% : {current_accuracy}, {current_f1_score}")
                # Making sure the script doesn't crash if the accuracy or f1 score is not found in the text file (in case of an error during the execution of the algorithm for example).
                current_accuracy = current_accuracy if current_accuracy is not None else "0"
                current_f1_score = current_f1_score if current_f1_score is not None else "0"

                m1 = ((current_accuracy + ",") * NUMBER_OF_PGRK_ALPHA_VALUES)[:-1] + "\n"
                m2 = ((current_f1_score + ",") * NUMBER_OF_PGRK_ALPHA_VALUES)[:-1] + "\n"

                m1 = f"{current_algo}," + m1
                m2 = f"{current_algo}," + m2

                with open(f"{folder_path}/{nb_dataset}_{current_percentage}_accuracy.csv", "a+") as f1:
                    f1.write(m1)
                with open(f"{folder_path}/{nb_dataset}_{current_percentage}_f1score.csv", "a+") as f1:
                    f1.write(m2)

                a -= 1 # Back to line just before percentage for continuous reading


            
            if 'pagerank' in current_algo: # 3 x 3 variants

                accu_list, f1score_list = [], []

                while '%' not in lines[a]:
                    
                    while 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "feature selection" not in lines[a].casefold():
                        
                        a += 1
                        if a >= len(lines):
                            break 

                    if a >= len(lines):
                        break 
                    

                    if 'accuracy' in lines[a].casefold():
                        accu_list.append(lines[a].split(':')[1].split(', ')[0])
                        f1score_list.append(lines[a].split(':')[1].split(', ')[1][:-1])

                    else:
                        break
                    
                    a += 1
                    if a >= len(lines):
                        break 
                
                # Making sure the script doesn't crash if the accuracy or f1 score is not found in the text file (in case of an error during the execution of the algorithm for example).
                accu_list = accu_list if accu_list is not None else ["0"] * NUMBER_OF_PGRK_ALPHA_VALUES
                f1score_list = f1score_list if f1score_list is not None else ["0"] * NUMBER_OF_PGRK_ALPHA_VALUES
                
                m1 = ",".join(accu_list) + "\n"
                m2 = ",".join(f1score_list) + "\n"

                m1 = f"{current_algo}," + m1
                m2 = f"{current_algo}," + m2

                with open(f"{folder_path}/{nb_dataset}_{current_percentage}_accuracy.csv", "a+") as f1:
                    f1.write(m1)
                with open(f"{folder_path}/{nb_dataset}_{current_percentage}_f1score.csv", "a+") as f1:
                    f1.write(m2)

                a -= 1 # Back to line just before percentage for continuous reading

        # Jump to next line        
        a += 1




def main():
    
    try:
        os.mkdir(BASE_FOLDER)
    except:
        pass

    for classifier in CLASSIFIERS_FOLDERS:
        try:
            os.mkdir(f"./{BASE_FOLDER}/{classifier}")
        except:
            pass

        # os.chdir(f"./{BASE_FOLDER}")
        results = {}
        txt_files = [f for f in os.listdir(f"{TXT_FOLDER}/{classifier}/") if f.endswith('.txt')]

        for txt_file in txt_files:
            # if txt_file not in ("dataset_4.txt", "dataset_7.txt"):

            # Extracting dataset number and gamma value from the filename using regex
            # Format : dataset_{dataset_nb}_classifier_{classifier}_gamma={gamma}.txt
            dataset_nb = regex.search(r'dataset_(\d+)_classifier', txt_file)
            if dataset_nb is not None:
                dataset_nb = dataset_nb.group(1)
            else:
                dataset_nb = txt_file.split('_')[1]
            
            gamma = regex.search(r'gamma=(\d*\.?\d+)', txt_file)
            if gamma is not None:
                gamma = gamma.group(1)
            else:
                gamma = txt_file.split('=')[1].split('.')[0] # format : dataset_{dataset_nb}_classifier_{classifier}_gamma={gamma}.txt


            try:
                os.mkdir(f"./{BASE_FOLDER}/{classifier}/dataset={dataset_nb}")
            except:
                pass

            try:
                os.mkdir(f"./{BASE_FOLDER}/{classifier}/dataset={dataset_nb}/gamma={gamma}")
            except:
                pass


            # for i in range(1, 11):
            #     accu_file = f"{BASE_FOLDER}/{classifier}/dataset={dataset_nb}/gamma={gamma}/{dataset_nb}_{str(10 * i)}_accuracy.csv"
            #     f1_file = f"{BASE_FOLDER}/{classifier}/dataset={dataset_nb}/gamma={gamma}/{dataset_nb}_{str(10 * i)}_f1score.csv"

                # open(accu_file, "w+")
                # open(f1_file, "w+")

            extract_data(f"{TXT_FOLDER}/{classifier}/{txt_file}", classifier, dataset_nb, gamma)
            print(f"\tDone extracting from {txt_file}")
        
        print(f"Done extracting from {classifier}.")

    print("Documents created successfully.")





if __name__ == "__main__":
    main()