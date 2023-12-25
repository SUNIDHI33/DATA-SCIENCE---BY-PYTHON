import pandas as pd
import matplotlib.pyplot as plt

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'


bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Check the column names
print(bank_train.columns)

response_counts = bank_train['y'].value_counts()
print(response_counts)

to_resample = bank_train.loc[bank_train['y'] == "yes"]
our_resample = to_resample.sample(n=841, replace=True)

bank_train_rebal = pd.concat([bank_train, our_resample])

response_counts_rebal = bank_train_rebal['y'].value_counts()
print(response_counts_rebal)
