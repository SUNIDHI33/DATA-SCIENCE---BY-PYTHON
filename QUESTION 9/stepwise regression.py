# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:29:01 2023

@author: Hp
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np

# Generate synthetic data
np.random.seed(42)
data = {'X1': np.random.rand(100),
        'X2': np.random.rand(100),
        'X3': np.random.rand(100),
        'Y': 2 * np.random.rand(100) + 3}

df = pd.DataFrame(data)

# Perform stepwise regression
def stepwise_selection(X, y):
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < 0.05:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval))

        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > 0.05:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break

    return included

X = df[['X1', 'X2', 'X3']]
y = df['Y']

selected_features = stepwise_selection(X, y)
print("\nSelected Features:")
print(selected_features)