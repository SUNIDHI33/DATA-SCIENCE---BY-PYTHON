from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter
# Assuming your data has a header row, set header=0
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Standardize the 'age' column
bank_train['age_z'] = stats.zscore(bank_train['age'])

# Find and filter outliers based on z-score
bank_train_outliers = bank_train.query('age_z > 3 | age_z < -3')

# Sort the DataFrame by 'age_z' in descending order
bank_train_sort = bank_train.sort_values('age_z', ascending=False)

# Display the top 15 rows of the sorted DataFrame for columns 'age' and 'marital'
top_outliers = bank_train_sort[['age', 'marital']].head(n=15)

# Print the result
print(top_outliers)
