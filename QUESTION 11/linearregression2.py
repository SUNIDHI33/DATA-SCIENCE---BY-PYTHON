# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 01:08:48 2023

@author: hp
"""


import pandas as pd
import statsmodels.api as sm

churn = pd.read_csv(r"\churn")
churn_ind = pd.get_dummies(churn['Churn'], drop_first = True)

X = pd.DataFrame(churn_ind)
X = sm.add_constant(X)
X.columns = ['const', 'Churn_True']

y = pd.DataFrame(churn[['CustServ Calls']])
poisreg01 = sm.GLM(y, X.astype(float), family = sm.families.Poisson()).fit()
print(poisreg01.summary())
