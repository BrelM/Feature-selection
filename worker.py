'''
worker.py

Orchestrate experiments across all feature-selection algorithms.

Changes vs original
-------------------
  • Algorithms 10-14 now iterate over damping factors [0.15, 0.50, 0.85]
    (passed via -p to main.py), matching the variation scheme used for
    the original PageRank (algos 8-9).

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
    0:  "Relief",
    1:  "ReliefF",
    2:  "Mutual information",
    3:  "Sequential feature selection",
    4:  "RFE-SVM",
    5:  "RFE-SVM-SFS",
    6:  "RIDGE",
    7:  "LASSO",
    8:  "PageRank",
    9:  "PageRank with deletion strategy",
    10: "UGFS",
    11: "PPRFS",
    12: "MGFS",
    13: "SGFS",
    14: "FSS-CPR",
}

# Three damping values tested for every graph-based algorithm (10-14)
DAMPING_VALUES = [0.15, 0.50, 0.85]

N_FEATURES_RANGE = 10   # 10%, 20%, …, 100%

# ─── CLI parsing ──────────────────────────────────────────────────────────────
try:
    cpts, args = getopt.getopt(sys.argv[1:], "d:", ["dataset="])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

classifier = 4     # Histogram gradient boosting (default)
dataset    = -1

for o, a in cpts:
    if o in ('-d', '--dataset'):
        try:
            dataset = int(a)
        except ValueError:
            print('dataset parameter must be an integer.')
            sys.exit(2)
    else:
        print(f"Unknown option: {o}")
        sys.exit(2)

# ─── Initialise report file ───────────────────────────────────────────────────
with open(f"reports/dataset_{dataset}.txt", "w+") as file:
    file.write(f"#################### Dataset : {DATASETS[dataset]} ####################\n\n")


# ─── Run all algorithms ───────────────────────────────────────────────────────
for algo in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:

    # skip multiclass datasets for Relief (binary only) and vice-versa
    if dataset in [1, 3, 4] and algo == 0:
        continue
    if dataset not in [1, 3, 4] and algo == 1:
        continue

    print(f"Feature selection algo : {ALGOS[algo]}")

    # ── RELIEF / RELIEFF (0-1) ────────────────────────────────────────────────
    if algo in [0, 1]:
        with open(f"reports/dataset_{dataset}.txt", "a+") as file:
            file.write(
                f"#################################################################\n"
                f"#################### Feature selection algo : {ALGOS[algo]} ####################\n"
            )

        for n_features in [(i + 1) / 10 for i in range(N_FEATURES_RANGE)]:
            print(f"  Number of features = {int(n_features * 100)}%")
            with open(f"reports/dataset_{dataset}.txt", "a+") as file:
                file.write(f"\nNumber of features = {int(n_features * 100)}%")

            for m in [5, 20, 50, 90]:
                os.system(
                    f"python main.py -d {dataset} -a {algo} -c {classifier} "
                    f"-p {m} -n {n_features}"
                )

    # ── RIDGE / LASSO (6-7) ───────────────────────────────────────────────────
    elif algo in [6, 7]:
        with open(f"reports/dataset_{dataset}.txt", "a+") as file:
            file.write(
                f"#################################################################\n"
                f"#################### Feature selection algo : {ALGOS[algo]} ####################\n"
            )

        for n_features in [(i + 1) / 10 for i in range(N_FEATURES_RANGE)]:
            print(f"  Number of features = {int(n_features * 100)}%")
            with open(f"reports/dataset_{dataset}.txt", "a+") as file:
                file.write(f"\nNumber of features = {int(n_features * 100)}%")

            for m in [0.00001, 0.01, 0.1, 1, 5, 20, 50]:
                os.system(
                    f"python main.py -d {dataset} -a {algo} -c {classifier} "
                    f"-p {m} -n {n_features}"
                )

    # ── PAGERANK ORIGINAL (8-9) ───────────────────────────────────────────────
    elif algo in [8, 9]:
        with open(f"reports/dataset_{dataset}.txt", "a+") as file:
            file.write(
                f"#################################################################\n"
                f"#################### Feature selection algo : {ALGOS[algo]} ####################\n"
            )

        for weighing_strat in ['corcoef', 'mi']:
            print(f"  Weighting strategy: {weighing_strat}")
            with open(f"reports/dataset_{dataset}.txt", "a+") as file:
                file.write(
                    f"\n#################### Graph weighting strategy: {weighing_strat} ####################\n\n"
                )

            for n_features in [(i + 1) / 10 for i in range(N_FEATURES_RANGE)]:
                print(f"    Number of features = {int(n_features * 100)}%")
                with open(f"reports/dataset_{dataset}.txt", "a+") as file:
                    file.write(f"\nNumber of features = {int(n_features * 100)}%")

                for m in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                    os.system(
                        f"python main.py -d {dataset} -a {algo} -c {classifier} "
                        f"-p {m} -n {n_features} -s {weighing_strat}"
                    )

    # ── NEW GRAPH-BASED ALGORITHMS (10-14) ────────────────────────────────────
    # Each algorithm is run with three damping factors: 0.15, 0.50, 0.85
    elif algo in [10, 11, 12, 13, 14]:
        with open(f"reports/dataset_{dataset}.txt", "a+") as file:
            file.write(
                f"#################################################################\n"
                f"#################### Feature selection algo : {ALGOS[algo]} ####################\n"
            )

        for n_features in [(i + 1) / 10 for i in range(N_FEATURES_RANGE)]:
            print(f"  Number of features = {int(n_features * 100)}%")
            with open(f"reports/dataset_{dataset}.txt", "a+") as file:
                file.write(f"\nNumber of features = {int(n_features * 100)}%")

            # ── three damping values ──────────────────────────────────────────
            for damping in DAMPING_VALUES:
                print(f"    damping = {damping}")
                os.system(
                    f"python main.py -d {dataset} -a {algo} -c {classifier} "
                    f"-n {n_features} -p {damping}"
                )

    # ── OTHER ALGORITHMS (2,3,4,5) ────────────────────────────────────────────
    else:
        with open(f"reports/dataset_{dataset}.txt", "a+") as file:
            file.write(
                f"#################################################################\n"
                f"#################### Feature selection algo : {ALGOS[algo]} ####################\n"
            )

        for n_features in [(i + 1) / 10 for i in range(N_FEATURES_RANGE)]:
            print(f"  Number of features = {int(n_features * 100)}%")
            with open(f"reports/dataset_{dataset}.txt", "a+") as file:
                file.write(f"\nNumber of features = {int(n_features * 100)}%")

            os.system(
                f"python main.py -d {dataset} -a {algo} -c {classifier} "
                f"-n {n_features}"
            )