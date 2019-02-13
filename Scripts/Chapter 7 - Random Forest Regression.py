# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:14:14 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

player_data = pd.read_csv('D:\Datasets\player.csv')

player_data.head()

features = player_data.iloc[:, 0:4].values  
labels = player_data.iloc[:, 4].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=200, random_state=0)  
rf_reg.fit(train_features, train_labels)  

predictions = rf_reg.predict( test_features)

comparison=pd.DataFrame({'Real':test_labels, 'Predictions':predictions})
print(comparison)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(test_labels, predictions))
print('MSE:', metrics.mean_squared_error(test_labels, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))