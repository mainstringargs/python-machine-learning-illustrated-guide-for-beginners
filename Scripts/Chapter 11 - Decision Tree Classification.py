# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:56:40 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris_data = pd.read_csv('D:\Datasets\iris_data.csv')

iris_data.head()

features= iris_data.iloc[:,0:4].values
labels= iris_data.iloc[:,4].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(train_features, train_labels)

predictions = dt_clf.predict( test_features)

comparison=pd.DataFrame({'Real':test_labels, 'Predictions':predictions})
print(comparison)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print(confusion_matrix(test_labels, predictions))  
print(classification_report(test_labels, predictions))  
print(accuracy_score(test_labels, predictions))