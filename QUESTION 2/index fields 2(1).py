import pandas as pd
import matplotlib.pyplot as plt

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter
# Assuming your data has a header row, set header=0
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Reset the index to use the default numeric index
bank_train.reset_index(drop=True, inplace=True)

# If you want to keep the existing index as a separate column, you can use:
# bank_train['old_index'] = bank_train.index

# Display the shape of the DataFrame
print(bank_train.shape)

# Display the first few rows of the DataFrame
print(bank_train.head())
