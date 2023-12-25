import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'
bank_train = pd.read_csv(file_path, delimiter=';', header=0)

# Select age for 'y = yes'
bt_age_y = bank_train[bank_train['y'] == "yes"]['age']

# Select age for 'y = no'
bt_age_n = bank_train[bank_train['y'] == "no"]['age']

# Plot histograms with overlay
plt.hist([bt_age_y, bt_age_n], bins=10, stacked=True, color=['blue', 'orange'])
plt.legend(['Response = Yes', 'Response = No'])
plt.title('Histogram of Age with Response Overlay')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
