

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

sales_train = pd.read_csv(r'C:\Users\Hp\Website Data Sets/clothing_sales_training.csv')
sales_test = pd.read_csv(r'C:\Users\Hp\Website Data Sets/clothing_sales_test.csv')

X = pd.DataFrame(sales_train[['Days', 'Web']])
X = sm.add_constant(X)
y = pd.DataFrame(sales_train['CC'])  # Assuming 'CC' is binary (0 or 1)
logreg01 = sm.Logit(y, X).fit()
print(logreg01.summary2())

X_test = pd.DataFrame(sales_test[['Days', 'Web']])
X_test = sm.add_constant(X_test)
y_test = pd.DataFrame(sales_test['CC'])  # Assuming 'CC' is binary (0 or 1)
logreg01_test = sm.Logit(y_test, X_test).fit()
print(logreg01_test.summary2())
