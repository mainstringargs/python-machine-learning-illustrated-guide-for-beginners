# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:17:56 2018

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

LDA = LinearDiscriminantAnalysis (n_components=3)  
train_features = LDA.fit_transform(train_features, train_labels)  
test_features = LDA.transform(test_features)  

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(train_features, train_labels)  
predictions = classifier.predict(test_features)  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print(confusion_matrix(test_labels, predictions))  
print(classification_report(test_labels, predictions))  
print(accuracy_score(test_labels, predictions))