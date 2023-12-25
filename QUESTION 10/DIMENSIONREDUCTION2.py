
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming correct file paths and file extensions
clothes_train = pd.read_csv(r'C:\Users\Hp\Website Data Sets/clothing_store_PCA_training')
clothes_test = pd.read_csv(r'C:\Users\Hp\Website Data Sets/clothing_store_PCA_test')

# Separate features and target variable
X = clothes_train.drop('Sales per Visit', axis=1)
print(X.corr())

# Standardize the features

# Apply PCA
pca = PCA(n_components=5)
principal_components = pca.fit_transform(X)

# Explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio for Each Principal Component:")
print(explained_variance_ratio)

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print("\nCumulative Explained Variance:")
print(cumulative_explained_variance)
