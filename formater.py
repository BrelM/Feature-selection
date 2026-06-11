'''
	formater.py

	This script contains functions to extract and format data from text files.

	By Alph@B, AKA Brel MBE
'''

import os

def extract_data(file_path, nb_dataset):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    dataset_name = lines[0].strip().split(': ')[1]
    results = []
    current_algo = None
    current_percentage = None
    current_meta_param = None
    current_features = []
    current_accuracy = None
    current_f1_score = None

    a = 0

    while a < len(lines):

        if "selection algo" in lines[a]:
            current_algo = lines[a].split(':')[1].replace('#', '').strip().casefold()

        if '%' in lines[a]:
            current_percentage = lines[a].split('=')[1].replace('%', '').strip().casefold()
            a += 1  # Go to next line

            # Single result algos (MI, SFS, RFE-SVM, RFE-SVM-SFS)
            if current_algo in ['mutual information', 'sequential feature selection', 'rfe-svm', 'rfe-svm-sfs']:

                while a < len(lines) and 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "selection algo :" not in lines[a].casefold():
                    a += 1

                if a < len(lines) and 'accuracy' in lines[a].casefold():
                    current_accuracy = lines[a].split(':')[1].split(', ')[0].strip()
                    current_f1_score = lines[a].split(':')[1].split(', ')[1].strip().rstrip('\n')

                    if current_accuracy is not None and current_f1_score is not None:
                        m1 = ((current_accuracy + ",") * 11)[:-1] + "\n"
                        m2 = ((current_f1_score + ",") * 11)[:-1] + "\n"

                        with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_accuracy.csv", "a+") as f1:
                            f1.write(m1)
                        with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_f1score.csv", "a+") as f1:
                            f1.write(m2)

            # Best result algos (Relief, ReliefF, Ridge, Lasso)
            elif current_algo in ['relief', 'relieff', 'ridge', 'lasso']:

                best_sum = 0.0
                best_accuracy = None
                best_f1_score = None

                while a < len(lines) and '%' not in lines[a]:

                    while a < len(lines) and 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "selection algo :" not in lines[a].casefold():
                        a += 1

                    if a < len(lines) and 'accuracy' in lines[a].casefold():
                        acc  = lines[a].split(':')[1].split(', ')[0].strip()
                        f1sc = lines[a].split(':')[1].split(', ')[1].strip().rstrip('\n')
                        try:
                            s = float(acc) + float(f1sc)
                        except ValueError:
                            s = 0.0
                        if s > best_sum:
                            best_sum      = s
                            best_accuracy = acc
                            best_f1_score = f1sc
                    else:
                        break

                    a += 1

                if best_accuracy is not None and best_f1_score is not None:
                    m1 = ((best_accuracy + ",") * 11)[:-1] + "\n"
                    m2 = ((best_f1_score + ",") * 11)[:-1] + "\n"

                    with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_accuracy.csv", "a+") as f1:
                        f1.write(m1)
                    with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_f1score.csv", "a+") as f1:
                        f1.write(m2)

                a -= 1

            # PageRank original (algos 8-9) + new graph-based algos (10-14)
            elif 'pagerank' in current_algo or current_algo in ['ugfs', 'pprfs', 'mgfs', 'sgfs', 'fss-cpr', 'prfs-imcc']:

                accu_list, f1score_list = [], []

                while a < len(lines) and '%' not in lines[a]:

                    while a < len(lines) and 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "selection algo :" not in lines[a].casefold():
                        a += 1

                    if a >= len(lines):
                        break

                    if 'accuracy' in lines[a].casefold():
                        accu_list.append(lines[a].split(':')[1].split(', ')[0].strip())
                        f1score_list.append(lines[a].split(':')[1].split(', ')[1].strip().rstrip('\n'))
                    else:
                        break

                    a += 1

                if accu_list:
                    # Padder à 11 colonnes si nécessaire pour homogénéité du CSV
                    while len(accu_list)    < 11: accu_list.append(accu_list[-1])
                    while len(f1score_list) < 11: f1score_list.append(f1score_list[-1])
                    accu_list    = accu_list[:11]
                    f1score_list = f1score_list[:11]
                    m1 = ",".join(accu_list) + "\n"
                    m2 = ",".join(f1score_list) + "\n"

                    with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_accuracy.csv", "a+") as f1:
                        f1.write(m1)
                    with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_f1score.csv", "a+") as f1:
                        f1.write(m2)

                a -= 1

        a += 1


def main():

    try:
        os.mkdir("./reports/Reports")
    except:
        pass

    os.chdir("./reports/Reports")

    results = {}
    txt_files = [f for f in os.listdir('../') if f.endswith('.txt')]

    for txt_file in txt_files:
        if txt_file not in ("dataset_4.txt", "dataset_7.txt"):

            try:
                os.mkdir("./" + txt_file[8])
            except:
                pass

            for i in range(1, 11):
                accu_file = f"{txt_file[8]}/{txt_file[8]}_{str(10 * i)}_accuracy.csv"
                f1_file   = f"{txt_file[8]}/{txt_file[8]}_{str(10 * i)}_f1score.csv"

                with open(accu_file, "w+") as f1:
                    pass
                with open(f1_file, "w+") as f2:
                    pass

            extract_data('../' + txt_file, txt_file[8])
            print(f"Done extracting from {txt_file}")

    print("Document created successfully.")

if __name__ == "__main__":
    main()