import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter

bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Assuming you have already loaded the dataset into bank_train

# Replace 999 with NaN in the 'age' column
bank_train['age'] = bank_train['age'].replace({999: np.NaN})

# Plot a histogram of the 'age' column
bank_train['age'].plot(kind='hist', title='Histogram of Age')

# Show the plot
plt.show()
