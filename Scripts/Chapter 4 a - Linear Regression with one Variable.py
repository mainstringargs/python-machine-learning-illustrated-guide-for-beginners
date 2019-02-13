# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 13:32:35 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

car_data = pd.read_csv('D:\Datasets\car_price.csv')

car_data.head()

car_data.describe()


plt.scatter(car_data['Year'], car_data['Price'])
plt.title("Year vs Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()


features= car_data.iloc[:,0:1].values
labels= car_data.iloc[:,1].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_features, train_labels)

print(lin_reg.coef_)

predictions = lin_reg.predict( test_features)

comparison=pd.DataFrame({'Real':test_labels, 'Predictions':predictions})
print(comparison)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(test_labels, predictions))
print('MSE:', metrics.mean_squared_error(test_labels, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
