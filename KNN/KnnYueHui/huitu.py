# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:46:03 2018

@author: sjdsb
"""
from numpy import *
import KnnYuehui
import matplotlib
import matplotlib.pyplot as plt
datingDataMat,datingLabels = KnnYuehui.file2matrix('datingTestSet.txt')
fig  = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),
           15.0*array(datingLabels))
plt.show()