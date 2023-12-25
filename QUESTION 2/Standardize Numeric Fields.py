from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter
# Assuming your data has a header row, set header=0
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Assuming 'age' is the column you want to standardize
bank_train['age_z'] = stats.zscore(bank_train['age'])
