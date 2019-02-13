# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:31:20 2018

@author: Mani
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

patient_data = pd.read_csv("D:/Datasets/patients.csv")

patient_data.head()

features= patient_data.iloc[:,0:3].values

labels= patient_data.iloc[:,3].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(features[:,1:2])
features[:,1:2] = imputer.transform(features[:,1:2])


from sklearn.preprocessing import LabelEncoder
labelencoder_features = LabelEncoder()
features[:,2]= labelencoder_features.fit_transform(features[:,2])


labels = labelencoder_features.fit_transform(labels)

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)
