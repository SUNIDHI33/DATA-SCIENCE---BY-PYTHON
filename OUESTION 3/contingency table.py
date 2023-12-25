import pandas as pd

# Load the dataset
file_path = r'C:\Users\Hp\Website Data Sets\bank-additional.csv'
df = pd.read_csv(file_path, delimiter=';')

# Assuming 'job' and 'y' are two columns for constructing the contingency table
contingency_table = pd.crosstab(df['job'], df['y'])

# Display the contingency table
print(contingency_table)
