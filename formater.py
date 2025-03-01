from docx import Document
from docx.shared import Pt
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
        # print(f"{lines[a].casefold()} {a}")

        if "selection algo" in lines[a]:
            current_algo = lines[a].split(':')[1].replace('#', '').strip().casefold()

        if '%' in lines[a]:
            current_percentage = lines[a].split('=')[1].replace('%', '').strip().casefold()
            a += 1 # Go to next line (there's a percentage on the current line)

            if current_algo in ['mutual information', 'sequential feature selection', 'rfe-svm']:
                
                while 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "feature selection" not in lines[a].casefold():
                    a += 1
            
                if 'accuracy' in lines[a].casefold():
                    current_accuracy = lines[a].split(':')[1].split(', ')[0]
                    current_f1_score = lines[a].split(':')[1].split(', ')[1]

                    m1 = ((current_accuracy + ",") * 11)[:-2] + "\n"
                    m2 = ((current_f1_score + ",") * 11)[:-2] + "\n"

                    with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_accuracy.csv", "a+") as f1:
                        f1.write(m1)
                    with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_f1score.csv", "a+") as f1:
                        f1.write(m2)



            if current_algo in ['relief', 'relieff', 'ridge', 'lasso']:

                metrics_sum, metrics = 0, []
                
                while '%' not in lines[a]:
                    
                    while 'accuracy' not in lines[a].casefold() and '%' not in lines[a].casefold() and "feature selection" not in lines[a].casefold():
                        a += 1
            
                    if 'accuracy' in lines[a].casefold():
                        metrics = [lines[a].split(':')[1].split(', ')[0], lines[a].split(':')[1].split(', ')[1]]

                        temp = float(metrics[0]) + float(metrics[1])
                        if temp > metrics_sum:
                            metrics_sum = temp
                            current_accuracy = metrics[0]
                            current_f1_score = metrics[1]

                    else:
                        break

                    a += 1


                m1 = ((current_accuracy + ",") * 11)[:-1] + "\n"
                m2 = ((current_f1_score + ",") * 11)[:-1] + "\n"

                with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_accuracy.csv", "a+") as f1:
                    f1.write(m1)
                with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_f1score.csv", "a+") as f1:
                    f1.write(m2)

                a -= 1 # Back to line just before percentage for continuous reading


            
            if 'pagerank' in current_algo: # 04 variants

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
                        f1score_list.append(lines[a].split(':')[1].split(', ')[1])

                    else:
                        break
                    a += 1

                m1 = ",".join(accu_list) + "\n"
                m2 = ",".join(f1score_list) + "\n"

                with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_accuracy.csv", "a+") as f1:
                    f1.write(m1)
                with open(nb_dataset + "/" + nb_dataset + "_" + current_percentage + "_f1score.csv", "a+") as f1:
                    f1.write(m2)

                a -= 1 # Back to line just before percentage for continuous reading
                
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
        if txt_file not in ("dataset_2.txt", "dataset_4.txt", "dataset_6.txt", "dataset_7.txt"):

            try:
                os.mkdir("./" + txt_file[8])
            except:
                pass

            for i in range(1, 11):
                accu_file = f"{txt_file[8]}/{txt_file[8]}_{str(10 * i)}_accuracy.csv"
                f1_file = f"{txt_file[8]}/{txt_file[8]}_{str(10 * i)}_f1score.csv"

                with open(accu_file, "w+") as f1:
                    pass

                with open(f1_file, "w+") as f2:
                    pass

            extract_data('../' + txt_file, txt_file[8])
            print(f"Done extracting from {txt_file}")

    print("Document created successfully.")

if __name__ == "__main__":
    main()