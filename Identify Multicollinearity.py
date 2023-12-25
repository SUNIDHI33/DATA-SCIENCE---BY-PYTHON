# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:07:14 2023

@auth
"""







import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as inf
cereals = pd.read_csv(r'C:\Users\Hp\Website Data Sets\cereals.csv')
X = pd.DataFrame(cereals[['Sugars', 'Fiber', 'Potass']])
pd.plotting.scatter_matrix(X)


X = X.dropna()
X = sm.add_constant(X)
vif = [inf.variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
vif_results = pd.DataFrame({'Variable': X.columns[1:], 'VIF': vif})

print(vif_results)





















