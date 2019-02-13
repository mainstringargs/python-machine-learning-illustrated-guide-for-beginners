# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 13:51:58 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

petrol_data = pd.read_csv('D:\Datasets\petrol_data.csv')

features= petrol_data.iloc[:,0:4].values
labels= petrol_data.iloc[:,4].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import PolynomialFeatures
poly_reg_feat = PolynomialFeatures(degree=2)
train_features_poly = poly_reg_feat.fit_transform(train_features)
test_features_poly = poly_reg_feat.transform(test_features)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features_poly = feature_scaler.fit_transform(train_features_poly)
test_features_poly = feature_scaler.transform(test_features_poly)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_features_poly, train_labels)

predictions = lin_reg.predict(test_features_poly)

comparison=pd.DataFrame({'Real':test_labels, 'Predictions':predictions})
print(comparison)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(test_labels, predictions))
print('MSE:', metrics.mean_squared_error(test_labels, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))

