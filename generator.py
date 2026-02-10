'''
generator.py

This script generates synthetic datasets for testing purposes.

By Alph@B, AKA Brel MBE
'''

import pandas as pd
import numpy as np

DATA_PATH = "../Data"

np.random.seed(42)

n = 10
df = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n),
    'income': np.random.randint(20000, 100000, size=n),
    'nb_loans': np.random.randint(1, 5, size=n),
    'nb_debts': np.random.randint(1, 10, size=n),
    'loan_amount': np.random.randint(1000, 50000, size=n),
    'risk_label': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
})

df.to_csv(DATA_PATH + '/mini_credit_risk/mini_credit_risk.csv', index=False)
