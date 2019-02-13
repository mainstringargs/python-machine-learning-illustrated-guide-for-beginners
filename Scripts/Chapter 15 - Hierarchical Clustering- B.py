# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:12:52 2018

@author: Mani
"""

import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np

customer_record = pd.read_csv('D:\Datasets\customer_records.csv') 

customer_record.head()  

dataset = customer_record.iloc[:, 3:5].values  

import scipy.cluster.hierarchy as hc

plt.figure(figsize=(12, 8))  
plt.title("Customer Clusters")  
link = hc.linkage(dataset, method='ward')
dendograms = hc.dendrogram(link)  

