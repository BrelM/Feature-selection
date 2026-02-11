# Feature Selection and Graph-Based Analysis Project

## Overview
This project is designed to execute and evaluate various feature selection (FS) techniques on multiple datasets. It includes graph-based methods, classifiers, and utilities for data processing and visualization. The project is modular, making it easy to extend or modify.

---

## Project Structure

### Root Files
- **`main.py`**: The entry point for executing feature selection algorithms. It handles parameterized execution, dataset loading, and algorithm selection.
- **`utils.py`**: Contains utility functions for data loading, graph building, and encoding.
- **`algorithms.py`**: Implements various feature selection algorithms.
- **`classifiers.py`**: Contains classifiers for evaluating selected features.
- **`pagerank.py`**: Implements PageRank-based feature selection algorithms.
- **`plotting.py`**: Handles graph visualization and plotting.
- **`generator.py`**: (If applicable) Generates synthetic datasets or graphs.
- **`formater.py`**: (If applicable) Formats datasets or results for consistency.
- **`worker.py`**: (If applicable) Manages parallel or batch processing tasks.

### Data and Reports
- **`reports/`**: Stores results of feature selection algorithms, including accuracy and F1-score metrics.
  - **`dataset_X.txt`**: Contains results for dataset `X`.
  - **`Reports/`**: Subdirectories for detailed CSV reports.
- **`pseudo_code_pagerank.txt`**: Contains pseudocode or notes for the PageRank algorithm.

---

## Key Components

### Datasets
The datasets are defined in `DATASETS_INFO` in `main.py` and `plotting.py`. Each dataset has metadata such as:
- Name
- Path
- Number of features
- Class index
- Separator
- Whether it contains categorical data

### Algorithms
Feature selection algorithms are defined in `ALGOS_INFO` in `main.py`. Examples include:
- Relief
- Mutual Information
- PageRank-based methods

### Classifiers
Classifiers for evaluating feature selection results are defined in `CLASSIFIERS_INFO` in `main.py`. Examples include:
- SVM
- Logistic Regression
- Decision Tree

### Graph Visualization
`plotting.py` is used to visualize graphs built from datasets. It supports:
- Circular layouts
- Edge weight labeling
- Filtering self-loops

---

## How to Use

### Running the Project with the worker script

It is recommended to use `worker.py` for executing tasks, as it simplifies batch processing and ensures efficient execution. Example:
```bash
python worker.py -d 2
```
#### Arguments for `worker.py`
- `-d` or `--dataset`: **Dataset index** (integer). Refers to the dataset to use, as defined in `DATASETS_INFO` in `main.py`.

This will automatically go through every feature selection algorithm avalaible (selecting from 10% to 100% of all features) and evaluate the results using every available classifiers.


### Running the Project with the main script

For more control, you can use `main.py` that enables you to specify the selection algorithm, the dataset, the classifier, the number of features to select and the graph weighting strategy.

```bash
python main.py -a 8 -d 2 -c 0 -n 0.5 -s corcoef
```
#### Arguments for `main.py`
- `-a` or `--algo`: **Algorithm index** (integer). Refers to the algorithm to use, as defined in `ALGOS_INFO` in `main.py`.
- `-d` or `--dataset`: **Dataset index** (integer). Refers to the dataset to use, as defined in `DATASETS_INFO` in `main.py`.
- `-c` or `--classif`: **Classifier index** (integer). Refers to the classifier to use for evaluation, as defined in `CLASSIFIERS_INFO` in `main.py`.
- `-n` or `--n_features`: **Percentage of features to select** (float). A value between 0 and 1 representing the proportion of features to retain.
- `-s`: **Graph weighting strategy** (string). Used for PageRank-based feature selection algorithms. Options include:
  - `corcoef`: Correlation coefficient-based weighting.
  - `mi`: Mutual information-based weighting.

For example, to run the PageRank algorithm on the Credit Risk dataset with SVM as the classifier, selecting 50% of features and using the correlation coefficient strategy, you would execute:
```bash
python main.py -a 8 -d 2 -c 0 -n 0.5 -s corcoef
```

### Adding or Updating Features

#### Adding a New Dataset
1. Update `DATASETS_INFO` in `main.py` and `plotting.py` with the new dataset's metadata.
2. Place the dataset file in the appropriate directory.

#### Adding a New Algorithm
1. Implement the algorithm in `algorithms.py` or a new script.
2. Add the algorithm to `ALGOS_INFO` in `main.py`.

#### Adding a New Classifier
1. Implement the classifier in `classifiers.py` or a new script.
2. Add the classifier to `CLASSIFIERS_INFO` in `main.py`.

#### Modifying Graph Visualization
1. Update `plotting.py` to customize graph layouts, edge filtering, or weight formatting.

---

## Notes
- Ensure all dependencies are installed (e.g., `matplotlib`, `networkx`).
- Follow the modular structure to maintain code readability and reusability.
- Use `reports/` to store and analyze results systematically.

---

## Contact
For questions or contributions, contact Alph@B (Brel MBE).