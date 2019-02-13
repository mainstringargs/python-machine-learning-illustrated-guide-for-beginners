# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 15:11:00 2018

@author: Mani
"""

import numpy as np

data = np.array([
    [1992,3000],  
    [1995,4000],
    [1998,4500],
    [1996,4200],
    [1999,4700],
    [1993,3500],
    [2001,5700],
    [2004,6000],
    [2008,6500],
    [2005,5800],
    [2007,6200],
    [2009,6700],])

import matplotlib.pyplot as plt

annots = range(1, 13)  
plt.figure(figsize=(12, 8))  
plt.subplots_adjust(bottom=0.1)  
plt.scatter(data[:,0],data[:,1], label='True Position')

for label, x, y in zip(annots, data[:, 0], data[:, 1]):  
    plt.annotate(
        label,
        xy=(x, y), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom')
plt.show() 

from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

annot = linkage(data, 'single')

marks = range(1, 13)

plt.figure(figsize=(12, 8))  
dendrogram(annot,  
            orientation='top',
            labels=marks,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()