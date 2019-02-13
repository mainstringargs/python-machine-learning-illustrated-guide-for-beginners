# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:46:01 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

banknote_data = pd.read_csv(r'D:\Datasets\banknote_data.csv')

banknote_data.shape

banknote_data.head()

features= banknote_data.iloc[:,0:4].values
labels= banknote_data.iloc[:,4].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

from sklearn.neighbors import KNeighborsClassifier  
knn_clf = KNeighborsClassifier(n_neighbors=3)  
knn_clf.fit(train_features, train_labels)

predictions = knn_clf.predict( test_features)

comparison=pd.DataFrame({'Real':test_labels, 'Predictions':predictions})
print(comparison)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print(confusion_matrix(test_labels, predictions))  
print(classification_report(test_labels, predictions))  
print(accuracy_score(test_labels, predictions))

rate_of_error = []


for i in range(1, 50):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_features, train_labels)
    predictions = knn.predict(test_features)
    rate_of_error.append(np.mean(predictions != test_labels))
    
    
plt.figure(figsize=(10, 5))  
plt.plot(range(1, 50), rate_of_error, color='green', linestyle='solid', marker='o',  
         markerfacecolor='red', markersize=10)
plt.title('K value vs Error')  
plt.xlabel('K Value')  
plt.ylabel('Error Value')  






