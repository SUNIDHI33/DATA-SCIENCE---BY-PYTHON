# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 02:08:18 2023

@author: Hp
"""

# 4_4 How to build C5.0 decision tree using python
from c50 import C5_0
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the C5.0 decision tree
clf = C5_0()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")