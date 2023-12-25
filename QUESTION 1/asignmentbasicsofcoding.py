# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:35:21 2023

@author: Hp
"""

# This is a comment

# importing packages in python
import pandas as pd
import numpy as np

#executing commands in pyhton
print("Hello World")

#getting data into python
bank_train = pd.read_csv( r'C:\Users\Hp\Website Data Sets/bank_marketing_training')

#saving output to pyhton
crosstab = pd.crosstab(bank_train['previous_outcome'], bank_train['response'])
print(crosstab)

print(bank_train.loc[0],
      bank_train.loc[0, 2, 3],
   ban[0:10])