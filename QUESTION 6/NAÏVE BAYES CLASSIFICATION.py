import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import statsmodels.tools.tools as stattools

# Load the training and test datasets
wine_tr = pd.read_csv(r"C:\Users\Hp\Website Data SetsS\wine_flag_training.csv", encoding='utf-8')
wine_test = pd.read_csv(r"C:\Users\Hp\Website Data Sets\wine_flag_test.csv", encoding='utf-8')

# Create a contingency table and plot for visualization
t1 = pd.crosstab(wine_tr['Type'], wine_tr['Alcohol_flag'])
t1['Total'] = t1.sum(axis=1)
t1.loc['Total'] = t1.sum()
t1

t1_plot = pd.crosstab(wine_tr['Alcohol_flag'], wine_tr['Type'])
t1_plot.plot(kind='bar', stacked=True)

# Encode categorical features using statsmodels
X_Alcohol_ind = np.array(wine_tr['Alcohol_flag'])
(X_Alcohol_ind, X_Alcohol_ind_dict) = stattools.categorical(X_Alcohol_ind, drop=True, dictnames=True)
X_Alcohol_ind = pd.DataFrame(X_Alcohol_ind)

X_Sugar_ind = np.array(wine_tr['Sugar_flag'])
(X_Sugar_ind, X_Sugar_ind_dict) = stattools.categorical(X_Sugar_ind, drop=True, dictnames=True)
X_Sugar_ind = pd.DataFrame(X_Sugar_ind)

# Concatenate the encoded features
X = pd.concat((X_Alcohol_ind, X_Sugar_ind), axis=1)

# Target variable
Y = wine_tr['Type']

# Train the Multinomial Naive Bayes classifier
nb_01 = MultinomialNB().fit(X, Y)

# Process the test dataset in a similar manner
X_Alcohol_ind_test = np.array(wine_test['Alcohol_flag'])
(X_Alcohol_ind_test, X_Alcohol_ind_dict_test) = stattools.categorical(X_Alcohol_ind_test, drop=True, dictnames=True)
X_Alcohol_ind_test = pd.DataFrame(X_Alcohol_ind_test)

X_Sugar_ind_test = np.array(wine_test['Sugar_flag'])
(X_Sugar_ind_test, X_Sugar_ind_dict_test) = stattools.categorical(X_Sugar_ind_test, drop=True, dictnames=True)
X_Sugar_ind_test = pd.DataFrame(X_Sugar_ind_test)

X_test = pd.concat((X_Alcohol_ind_test, X_Sugar_ind_test), axis=1)

# Predict the wine types for the test dataset
Y_predicted = nb_01.predict(X_test)

# Create a contingency table for predicted vs. actual values
ypred = pd.crosstab(wine_test['Type'], Y_predicted, rownames=['Actual'], colnames=['Predicted'])
ypred['Total'] = ypred.sum(axis=1)
ypred.loc['Total'] = ypred.sum()

# Display the contingency table
print(ypred)
