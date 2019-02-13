# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:20:18 2018

@author: Mani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

redwine_data = pd.read_csv(r'D:\Datasets\redwine_data.csv', sep=';')

redwine_data.head()

features= redwine_data.iloc[:,0:11].values
labels= redwine_data.iloc[:,11].values

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0, random_state = 0)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)


from sklearn.ensemble import RandomForestClassifier  
rf_clf = RandomForestClassifier(n_estimators=500, random_state=0)  

from sklearn.model_selection import cross_val_score  
rf_accuracies = cross_val_score(estimator=rf_clf, X=train_features, y =train_labels, cv=5)  

print(rf_accuracies)  

print(rf_accuracies.mean())  

print(rf_accuracies.std())  


########### Grid Search #################3

param = {  
    'n_estimators': [100, 250, 500, 750, 1000],
    'warm_start': ['True', 'False'],  
    'criterion': ['entropy', 'gini']
   
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=rf_clf,  
                     param_grid=param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

grid_search.fit(train_features, train_labels)  

optimal_parameters = grid_search.best_params_  
print(optimal_parameters)  

optimal_results = grid_search.best_score_  
print(optimal_results)

