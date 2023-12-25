import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Replace the file path with the actual path to your dataset file
file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'

# Load the dataset into a DataFrame with the correct delimiter
# Assuming your data has a header row, set header=0
bank_train = pd.read_csv(file_path, delimiter=';', header=0)
print(bank_train)

# Assuming 'education' is the column you want to reexpress
dict_edu = {
    "education_numeric": {
        "illiterate": 0,
        "basic.4y": 4,
        "basic.6y": 6,
        "basic.9y": 9,
        "high.school": 12,
        "professional.course": 12,  # Corrected the name
        "university.degree": 16,
        "unknown": np.NaN
    }
}

# Replace values in the 'education' column and create a new 'education_numeric' column
bank_train.replace(dict_edu, inplace=True)
