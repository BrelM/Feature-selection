# Feature Selection and Graph-Based Analysis Project
# Projet de Sélection d'Attributs par Graphes

---

## 🇬🇧 English Version

### Overview
This project implements and evaluates various feature selection techniques on multiple datasets. It includes classical methods (Relief, MI, RFE...), graph-based PageRank methods, and our novel **PRFS-IMCc** algorithm. Results are saved as CSV reports for analysis.

---

### Project Structure

| File | Role |
|------|------|
| `main.py` | CLI entry point — runs one algorithm on one dataset |
| `worker.py` | Batch launcher — runs all algorithms on a dataset |
| `algorithms.py` | Implementation of all feature selection algorithms |
| `pagerank.py` | PageRank implementation (standard and personalized) |
| `classifiers.py` | Classifiers for evaluating selected features |
| `utils.py` | Data loading, encoding, utilities |
| `formater.py` | Extracts and formats results into CSV files |
| `plotting.py` | Graph visualization |

---

### Available Algorithms

| Index | Name | Description |
|-------|------|-------------|
| 0 | ReliefF | Relief based on distances between instances |
| 1 | Mutual Information | Mutual information with the class |
| 2 | RFE-SVM | Recursive Feature Elimination with SVM |
| 3 | RFE-SVM-SFS | RFE-SVM + Sequential Forward Selection |
| 5 | Ridge | Ridge regression |
| 6 | Lasso | Lasso regression |
| 7 | SFS | Sequential Feature Selection |
| 8 | PageRank | PageRank on dependency graph (undirected) |
| 9 | PageRank + Deletion | PageRank with deletion strategy |
| 10 | UGFS | Unsupervised Graph-based Feature Selection (Henni et al. 2018) |
| 11 | PPRFS | Personalized PageRank Feature Selection |
| 12 | MGFS | Multi-Graph Feature Selection |
| 13 | SGFS | Supervised Graph-based Feature Selection |
| 14 | FSS-CPR | Feature Selection via Subspace and PageRank |
| **15** | **PRFS-IMCc** | **Our method — PageRank + Conditional Mutual Information of Class** |

---

### Our Method: PRFS-IMCc (algo 15)

#### Principle
PRFS-IMCc is a supervised feature selection method based on an **oriented weighted graph** and **Personalized PageRank**.

#### Oriented graph construction
For each pair (Aᵢ, Aⱼ), the weight of the oriented edge is:

```
dependence(Aᵢ → Aⱼ) = (1 - γ) × measure_var(Aᵢ, Aⱼ) + γ × IMCc(Aᵢ, Aⱼ)
```

Where:
- **measure_var** ∈ {`corr` (Pearson), `mi` (Mutual Information), `theil` (Theil U)}
- **IMCc(Aᵢ, Aⱼ)** = I(Aⱼ ; Class | Aᵢ) — how much Aⱼ tells about the class given Aᵢ
- **γ** ∈ {0.2, 0.5, 0.8} — balance between raw dependence and class information

#### Parameter `-p gamma:damping`
The `-p` argument encodes **two parameters** separated by `:`:

```
-p  0.5 : 0.85
     ↑      ↑
   gamma  damping
```

- **gamma (γ)**: weight of IMCc in the edge weight formula
  - γ = 0.2 → graph dominated by raw dependence (Corr/MI/Theil)
  - γ = 0.5 → balanced
  - γ = 0.8 → graph dominated by class information (IMCc)

- **damping (α)**: PageRank damping factor
  ```
  PR(t+1) = α × M^T × PR(t)  +  (1-α) × v
              ↑                      ↑
        follow graph edges     teleport toward d(Aᵢ) = IM(Aᵢ, Class)
  ```
  - α = 0.15 → strong pull toward initial relevance scores
  - α = 0.85 → mostly follows the graph structure

#### Greedy selection with Personalized PageRank
1. Compute d(Aᵢ) = IM(Aᵢ, Class) / Σ IM(Aₖ, Class) — personalization vector
2. Run Personalized PageRank → select the node with the highest score
3. Remove that node from the graph, update scores via Theil U
4. Repeat until the desired number of features is reached

#### Parameters summary

| Parameter | Values | Description |
|-----------|--------|-------------|
| `-s` | `corr`, `mi`, `theil` | Graph weighting measure |
| `-p` | `γ:damping` e.g. `0.5:0.85` | Gamma and damping factor |
| `-n` | `0.2`, `0.3` | Percentage of features to select |

---

### How to Use

#### Run a single test
```bash
# PRFS-IMCc: gamma=0.5, damping=0.85, 20% features, dataset 1, classifier 4, measure=theil
python main.py -d 1 -a 15 -c 4 -n 0.2 -p 0.5:0.85 -s theil

# Classic PageRank: dataset 2, classifier 0, 50% features, MI weighting
python main.py -a 8 -d 2 -c 0 -n 0.5 -s mi

# UGFS: dataset 1, classifier 4, 30% features
python main.py -d 1 -a 10 -c 4 -n 0.3 -p 0.85
```

#### `main.py` arguments

| Argument | Description |
|----------|-------------|
| `-a` | Algorithm index (see table above) |
| `-d` | Dataset index |
| `-c` | Classifier index |
| `-n` | Proportion of features to select (e.g. `0.2` = 20%) |
| `-s` | Graph weighting strategy: `corr`, `mi`, `theil` |
| `-p` | Algorithm parameter(s). For algo 15: `gamma:damping` (e.g. `0.5:0.85`) |

#### Run all algorithms on a dataset
```bash
python worker.py -d 1
```
Runs all algorithms (0–15) on dataset 1 with all parameter combinations.

For **PRFS-IMCc (algo 15)**, `worker.py` automatically iterates over:
- `n_features` ∈ {20%, 30%}
- `measure` ∈ {corr, mi, theil}
- `gamma` ∈ {0.2, 0.5, 0.8}
- `damping` ∈ {0.15, 0.50, 0.85}

Total: **54 combinations** per dataset.

#### Extract results as CSV
```bash
python formater.py
```
Generates CSV files in `reports/Reports/X/` where X is the dataset index.

Each `X_Y_accuracy.csv` file contains:
- One **row per algorithm**
- One **column per parameter combination**

---

### Installation
```bash
pip install -r requirements.txt
```

---

### References
- **UGFS**: Henni et al., *Unsupervised graph-based feature selection via subspace and pagerank centrality*, Expert Systems With Applications 114 (2018) 46–53.
- **PPRFS**: Zhu et al., IEEE 2019.
- **PRFS-IMCc**: Our contribution — PageRank Feature Selection with Conditional Mutual Information of Class.

---

### Contact
For questions or contributions: Alph@B (Brel MBE).

---
---

## 🇫🇷 Version Française

### Vue d'ensemble
Ce projet implémente et évalue différentes techniques de sélection d'attributs sur plusieurs jeux de données. Il inclut des méthodes classiques (Relief, MI, RFE...), des méthodes basées sur les graphes et PageRank, ainsi que notre nouvel algorithme **PRFS-IMCc**. Les résultats sont sauvegardés sous forme de rapports CSV pour analyse.

---

### Structure du projet

| Fichier | Rôle |
|---------|------|
| `main.py` | Point d'entrée CLI — exécute un algo sur un dataset |
| `worker.py` | Lance tous les algos sur un dataset en batch |
| `algorithms.py` | Implémentation de tous les algorithmes de sélection |
| `pagerank.py` | Implémentation du PageRank (standard et personnalisé) |
| `classifiers.py` | Classifieurs pour évaluer les features sélectionnées |
| `utils.py` | Chargement des données, encodage, utilitaires |
| `formater.py` | Extraction et mise en forme des résultats en CSV |
| `plotting.py` | Visualisation des graphes |

---

### Algorithmes disponibles

| Index | Nom | Description |
|-------|-----|-------------|
| 0 | ReliefF | Relief basé sur les distances entre instances |
| 1 | Mutual Information | Information mutuelle avec la classe |
| 2 | RFE-SVM | Recursive Feature Elimination avec SVM |
| 3 | RFE-SVM-SFS | RFE-SVM + Sequential Forward Selection |
| 5 | Ridge | Régression Ridge |
| 6 | Lasso | Régression Lasso |
| 7 | SFS | Sequential Feature Selection |
| 8 | PageRank | PageRank sur graphe de dépendances (non orienté) |
| 9 | PageRank + Deletion | PageRank avec stratégie de suppression |
| 10 | UGFS | Unsupervised Graph-based Feature Selection (Henni et al. 2018) |
| 11 | PPRFS | Personalized PageRank Feature Selection |
| 12 | MGFS | Multi-Graph Feature Selection |
| 13 | SGFS | Supervised Graph-based Feature Selection |
| 14 | FSS-CPR | Feature Selection via Subspace and PageRank |
| **15** | **PRFS-IMCc** | **Notre méthode — PageRank + Information Mutuelle Conditionnelle de Classe** |

---

### Notre méthode : PRFS-IMCc (algo 15)

#### Principe
PRFS-IMCc est une méthode supervisée de sélection d'attributs basée sur un **graphe orienté** pondéré et le **PageRank Personnalisé**.

#### Construction du graphe orienté
Pour chaque paire (Aᵢ, Aⱼ), le poids de l'arête orientée Aᵢ → Aⱼ est :

```
dépendance(Aᵢ → Aⱼ) = (1 - γ) × mesure_var(Aᵢ, Aⱼ) + γ × IMCc(Aᵢ, Aⱼ)
```

Où :
- **mesure_var** ∈ {`corr` (Pearson), `mi` (Information Mutuelle), `theil` (Theil U)}
- **IMCc(Aᵢ, Aⱼ)** = I(Aⱼ ; Classe | Aᵢ) — combien Aⱼ apporte sur la classe sachant Aᵢ
- **γ** ∈ {0.2, 0.5, 0.8} — équilibre entre dépendance brute et information de classe

#### Le paramètre `-p gamma:damping`
L'argument `-p` encode **deux paramètres** séparés par `:` :

```
-p  0.5 : 0.85
     ↑      ↑
   gamma  damping
```

- **gamma (γ)** : poids de IMCc dans la formule de pondération des arêtes
  - γ = 0.2 → graphe dominé par la mesure brute (Corr/MI/Theil)
  - γ = 0.5 → équilibre parfait entre mesure brute et IMCc
  - γ = 0.8 → graphe dominé par l'information de classe (IMCc)

- **damping (α)** : facteur d'amortissement du PageRank
  ```
  PR(t+1) = α × M^T × PR(t)  +  (1-α) × v
              ↑                      ↑
        suivre les arêtes       se rappeler d(Aᵢ) = IM(Aᵢ, Classe)
        du graphe               (importance initiale vis-à-vis de la classe)
  ```
  - α = 0.15 → forte attraction vers les scores de pertinence initiaux
  - α = 0.85 → suit principalement la structure du graphe

#### Sélection greedy avec PageRank Personnalisé
1. Calculer d(Aᵢ) = IM(Aᵢ, Classe) / Σ IM(Aₖ, Classe) — vecteur de personnalisation
2. Lancer le PageRank Personnalisé → sélectionner le nœud avec le plus grand score
3. Retirer ce nœud du graphe, mettre à jour les scores via Theil U
4. Répéter jusqu'à avoir le nombre voulu d'attributs

#### Résumé des paramètres

| Paramètre | Valeurs | Description |
|-----------|---------|-------------|
| `-s` | `corr`, `mi`, `theil` | Mesure de pondération du graphe |
| `-p` | `γ:damping` ex: `0.5:0.85` | Gamma et facteur d'amortissement |
| `-n` | `0.2`, `0.3` | Pourcentage de features à sélectionner |

---

### Comment utiliser

#### Lancer un seul test
```bash
# PRFS-IMCc : gamma=0.5, damping=0.85, 20% features, dataset 1, classifieur 4, mesure=theil
python main.py -d 1 -a 15 -c 4 -n 0.2 -p 0.5:0.85 -s theil

# PageRank classique : dataset 2, classifieur 0, 50% features, pondération MI
python main.py -a 8 -d 2 -c 0 -n 0.5 -s mi

# UGFS : dataset 1, classifieur 4, 30% features
python main.py -d 1 -a 10 -c 4 -n 0.3 -p 0.85
```

#### Arguments de `main.py`

| Argument | Description |
|----------|-------------|
| `-a` | Index de l'algorithme (voir tableau ci-dessus) |
| `-d` | Index du dataset |
| `-c` | Index du classifieur |
| `-n` | Proportion de features à sélectionner (ex: `0.2` = 20%) |
| `-s` | Stratégie de pondération : `corr`, `mi`, `theil` |
| `-p` | Paramètre(s). Pour algo 15 : `gamma:damping` (ex: `0.5:0.85`) |

#### Lancer tous les algos sur un dataset
```bash
python worker.py -d 1
```
Lance tous les algorithmes (0–15) sur le dataset 1 avec toutes les combinaisons de paramètres.

Pour **PRFS-IMCc (algo 15)**, `worker.py` itère automatiquement sur :
- `n_features` ∈ {20%, 30%}
- `measure` ∈ {corr, mi, theil}
- `gamma` ∈ {0.2, 0.5, 0.8}
- `damping` ∈ {0.15, 0.50, 0.85}

Soit **54 combinaisons** par dataset.

#### Extraire les résultats en CSV
```bash
python formater.py
```
Génère les fichiers CSV dans `reports/Reports/X/` (X = index du dataset).

Chaque fichier `X_Y_accuracy.csv` contient :
- Une **ligne par algorithme**
- Une **colonne par combinaison de paramètres**

---

### Installation des dépendances
```bash
pip install -r requirements.txt
```

---

### Notes importantes
- Le dossier `reports/` est ignoré par Git — les résultats sont générés localement.
- Pour ajouter un nouveau dataset : mettre à jour `DATASETS_INFO` dans `main.py`.
- Pour ajouter un nouvel algorithme : implémenter dans `algorithms.py`, ajouter dans `ALGOS` et `ALGOS_INFO` dans `main.py` et `worker.py`, et ajouter le nom dans `formater.py`.

---

### Références
- **UGFS** : Henni et al., *Unsupervised graph-based feature selection via subspace and pagerank centrality*, Expert Systems With Applications 114 (2018) 46–53.
- **PPRFS** : Zhu et al., IEEE 2019.
- **PRFS-IMCc** : Notre contribution — PageRank Feature Selection avec Information Mutuelle Conditionnelle de Classe.

---

### Contact
Pour questions ou contributions : Alph@B (Brel MBE).