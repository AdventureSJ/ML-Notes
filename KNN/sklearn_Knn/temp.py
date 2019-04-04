# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
函数说明：将32*32的二进制图像转换为1*1024向量。
Parameter:
    filename - 文件名
Return：
    returnVect - 返回的二进制图像的1*1024向量
Modify:
    2018/9/5
"""
def img2vector(filename):
    #创建1*1024的零向量初始化返回向量矩阵
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读取每一行数据
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

"""
函数说明：手写数字分类测试
Parameters:
    无
Return：
    无
Modify：
    2018/9/5
"""
def handwritingClassTest():
    #训练集的labels
    hwlabel=[]
    #返回训练集目录下的文件名
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    #初始化训练集
    trainingMat = np.zeros((m,1024))
    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #训练集真实labels
        classNumber = int(fileNameStr.split('_')[0])
        hwlabel.append(classNumber)
        #将每一个文件的1*1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s'%(fileNameStr))
    #构建KNN分类器
    neigh = KNN(n_neighbors = 3,algorithm = 'auto')
    #拟合模型
    neigh.fit(trainingMat,hwlabel)
    #返回测试集下的文件列表
    testFileList = listdir('testDigits')
    #检测错误数据
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        testNameStr = testFileList[i]
        classNumber = int(testNameStr.split('_')[0])
        vectorTest = img2vector('testDigits/%s'%(testNameStr))
        classiFileResult = neigh.predict(vectorTest)
        print("分类返回结果为%d\t真实结果为%d"%(classiFileResult,classNumber))
        if(classiFileResult!=classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%"%(errorCount,errorCount/mTest*100))
    
    
    
if __name__ =='__main__':
    handwritingClassTest()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        















