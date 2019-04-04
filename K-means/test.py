# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:26:38 2018

@author: sjdsb
"""
import kMeans
import numpy as np
import matplotlib.pyplot as plt
#dataMat= np.mat(kMeans.loadDataSet('testSet2.txt'))
#cenList,myNewAssment = kMeans.biKmeans(dataMat,3)
#plt.scatter(dataMat[:,0].tolist(),dataMat[:,1].tolist())
#
        #plt.scatter(PtsInClust[:,0].tolist(),PtsInClust[:,1].tolist(),
                    #marker='v',color='k',label='3',s=12)
        #plt.scatter(myCentroids[i,0],myCentroids[i,1],color='k',marker='+',s=180)
kMeans.clusterClubs(5)

