# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:33:39 2023

@author: Hp
"""

import numpy as np
import scipy.stats as stats

group1_data = np.random.normal(0, 1, 100)
group2_data = np.random.normal(1, 1, 100)
mean_group1 = np.mean(group1_data)
mean_group2 = np.mean(group2_data)
std_dev_group1 = np.std(group1_data, ddof=1)
std_dev_group2 = np.std(group2_data, ddof=1)

confidence_level = 0.95

std_error_diff = np.sqrt((std_dev_group1**2 / len(group1_data)) + (std_dev_group2**2 / len(group2_data)))
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=len(group1_data) + len(group2_data) - 2) * std_error_diff

confidence_interval = (mean_group1 - mean_group2 - margin_of_error, mean_group1 - mean_group2 + margin_of_error)

if confidence_interval[0] <= 0 <= confidence_interval[1]:
    print("The difference is not statistically significant.")
else:
    print("The difference is statistically significant.")