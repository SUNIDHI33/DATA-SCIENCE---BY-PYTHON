import pandas as pd
from sklearn.model_selection import train_test_split

file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Use train_test_split to split the data into training and testing sets
bank_train, bank_test = train_test_split(bank_train, test_size=0.25, random_state=7)

# Print the shapes of the original, training, and testing sets
print("Original shape:", bank_train.shape)
print("Training set shape:", bank_train.shape)
print("Testing set shape:", bank_test.shape)
