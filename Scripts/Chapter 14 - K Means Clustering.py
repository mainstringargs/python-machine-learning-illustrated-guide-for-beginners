# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:07:27 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

car_data = pd.read_csv('D:\Datasets\car_price.csv')

car_data.head()

plt.scatter(car_data['Year'], car_data['Price'])
plt.title("Year vs Price")
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()

from sklearn.cluster import KMeans
km_clus = KMeans(n_clusters=3)
km_clus.fit(car_data)

print(km_clus.cluster_centers_)

print(km_clus.labels_)

plt.scatter(car_data['Year'], car_data['Price'], c = km_clus.labels_, cmap='rainbow')

plt.scatter(car_data['Year'], car_data['Price'], c = km_clus.labels_, cmap='rainbow')
plt.scatter(km_clus.cluster_centers_[:,0],km_clus.cluster_centers_[:,1], color='yellow')
