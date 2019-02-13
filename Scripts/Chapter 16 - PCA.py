# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:15:36 2018

@author: Mani
"""

import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np

banknote_data = pd.read_csv(r'D:\Datasets\banknote_data.csv')

banknote_data.head()

features= banknote_data.iloc[:,0:4].values
labels= banknote_data.iloc[:,4].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

from sklearn.decomposition import PCA

pca = PCA(4)  
train_features = pca.fit_transform(train_features)  
test_features = pca.transform(test_features)  

exp_var = pca.explained_variance_ratio_ 
print(exp_var)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=50, random_state=0)  
rf_clf.fit(train_features, train_labels)

predictions = rf_clf .predict( test_features)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print(confusion_matrix(test_labels, predictions))  
print(classification_report(test_labels, predictions))  
print(accuracy_score(test_labels, predictions))
