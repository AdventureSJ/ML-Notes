# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:43:41 2018

@author: jacksong1996
"""
from numpy import *
import matplotlib.pyplot as plt
import random

def loadDataSet():
    '''
    函数说明：读取数据集
    Returns: dataMat - 包含了数据特征值的矩阵
             dataLabel - 数据集的标签
    '''
    dataMat = [];dataLabel = []
    fr = open("testSet.txt")
    fileLines = fr.readlines()
    
    for line in fileLines:
        lineArr = line.strip().split('\t')
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        dataLabel.append(int(lineArr[2]))
        
    return dataMat,dataLabel

def sigmoid(inx):
    '''
    函数说明：sigmoid函数
    Parameters: inx - 经历一次前向传播还没有激活的值
    Returns:    激活后的值
    '''
    return 1.0/(1+exp(-inx))

def gradAscent(dataMatIn,classLabels):
    #将输入数据集转换为矩阵
    dataMatrix = mat(dataMatIn)
    LabelsMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    weights = ones((n,1))       #初始化系数矩阵，因为数据集为m行n列，所以权重矩阵为n行
    maxIter = 500       #迭代次数
    alpha = 0.001       #步长，也可以称作学习率
    for i in range(maxIter):
        z = sigmoid(dataMatrix * weights)
        error = LabelsMat - z       #计算误差
        weights = weights + alpha * dataMatrix.transpose() * error      #更新权重矩阵
    return weights

def stocGradAscent(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    Weight = zeros((numIter,n))
    weights = ones(n)
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randindex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randindex]*weights))
            error = classLabels[randindex] - h
            weights = weights + [alpha*error*k for k in dataMatrix[randindex]]
            del(dataIndex[randindex])
        Weight[i,:] = weights
            
    return weights

def plotBestFit(weights):
    dataMat,dataLabel = loadDataSet()
    dataArray = array(dataMat)
    xcord1 = [];ycord1 = []
    xcord2 = [];ycord2 = []
    n = shape(dataArray)[0]
    for i in range(n):
        if int(dataLabel[i]) == 0:
            xcord1.append(dataArray[i,1])
            ycord1.append(dataArray[i,2])
        else:
            xcord2.append(dataArray[i,1])
            ycord2.append(dataArray[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,color = 'red',marker = 's',s = 30)
    ax.scatter(xcord2,ycord2,color = 'blue',s = 30)
    x = arange(-3.0,3.0,0.1)
    #y = (-weights[0,0] - weights[1,0]*x)/weights[2,0]      #0=w0X0+w1X1+w2X2
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
    
def plotWeight(weight):
    n = shape(weight)[0]
    x = list(arange(0,n,1))
    fig = plt.figure()
    
    ax = fig.add_subplot(3,1,1)
    ax.plot(x,weight[:,0])
    plt.ylabel("X0")
    
    ax = fig.add_subplot(3,1,2)
    ax.plot(x,weight[:,1])
    plt.ylabel("X1")
    
    ax = fig.add_subplot(3,1,3)
    ax.plot(x,weight[:,2])
    plt.ylabel("X2")
    plt.xlabel("numIter")
    plt.show()

def classifyVector(inx,weights):
    '''
    函数说明：分类函数
    Parameter：inx - 输入的特征向量
               weights - 回归系数
    return: 类别标签
    '''
    prob = sigmoid(sum(inx*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0    

def colicTest():
    '''
    函数说明：打开测试集和训练集，训练数据预测错误率
    Parameter：无
    Retrurn：errorRate - 错误概率
    '''
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainList = [];trainLabels = []
    for line in frTrain.readlines():
        lineArr = []
        currLine = line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainList.append(lineArr)
        trainLabels.append(float(currLine[21]))     #处理训练集数据
        
    weight = stocGradAscent(array(trainList),trainLabels,500)  #随机梯度上升得到回归系数
    errorCount = 0;numTest = 0.0
    for line in frTest.readlines():
        lineArr = []
        numTest += 1.0
        currLine = line.strip().split('\t')
        for i in range(21):
            lineArr.append(float(currLine[i]))    #处理测试集数据   
        if int(classifyVector ( array(lineArr), weight)) != int( currLine[21] ):
            errorCount += 1
    errorRate = float(errorCount) / numTest       #计算错误率
    print("the test errorrate is %f" % errorRate)
    return errorRate

def multiTest():
    '''
    函数说明：计算迭代numtests次后的错误率
    '''
    
    numtests = 10;errorsum = 0.0
    for k in range(numtests):
        errorsum += colicTest()
    print("after %d iterations the average error rate is:%f" % (numtests,errorsum/float(numtests)))
    
        
if __name__=='__main__':
    #dataArr,labelMat = loadDataSet()
    #weights,Weight = stocGradAscent(dataArr,labelMat)
    #plotBestFit(weights)
    #plotWeight(Weight)
    multiTest()






