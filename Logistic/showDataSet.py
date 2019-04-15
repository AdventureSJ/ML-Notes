# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:33:08 2018

@author: jacksong1996
"""
import matplotlib.pyplot as plt
import numpy as np

def showDataSet():
    fr = open("testSet.txt")
    filelines = fr.readlines()
    dataMat = np.zeros((len(filelines),2))
    ClassLabelColors = []
    index = 0
    for each in filelines:
        
        line = each.strip()
        ListFromline = line.split('\t')
        dataMat[index,:] = ListFromline[0:2]
        if ListFromline[-1] == '1':
            ClassLabelColors.append('red')
        else:
            ClassLabelColors.append('black')
        index += 1
        
    plt.scatter(x=dataMat[:,0],y=dataMat[:,1],color=ClassLabelColors,s=15,alpha=0.5)   

if __name__ == '__main__':
    showDataSet()
    
    
        
    
