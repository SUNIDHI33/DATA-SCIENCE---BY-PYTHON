# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:24:25 2023

@author: Hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 12:14:57 2023

@author: hp
"""

# 7_1 Neural networks
# Demonstrate application of nueral networks.

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

model = models.Sequential()
model.add(layers.Dense(1, input_dim=1))
model.compile(optimizer='sgd', loss='mean_squared_error')
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

model.fit(X, y, epochs=1000, verbose=0)

new_data = np.array([6], dtype=float)
prediction = model.predict(new_data)

print("Prediction for input 6:", prediction[0, 0])
