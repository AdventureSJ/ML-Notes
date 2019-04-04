# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 20:06:33 2018

@author: sjdsb
"""

from math import log
import operator

"""
函数说明：计算给定数据集的经验熵
Parameters:
    dataSet - 数据集
Returns：
    shannonEnt - 经验熵（香农熵）
Modify:
    2018/9/6
"""

def calcShannonEnt(dataSet):
    numdataRows = len(dataSet)          #返回数据集行数，即样本个数
    labelCounts = {}                    #用于计算数据集最后一列标签向量
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  #获得数据集最后一列标签的对应的字典
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numdataRows
        shannonEnt -= prob*log(prob,2) #按照经验熵公式计算经验熵
    return shannonEnt


"""
函数说明：创建测试数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 特征标签
Modify：
    2018/9/6
"""

def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄','有工作','有自己的房子','信贷情况']
    return dataSet,labels


"""
函数说明：按照给定特征划分数据集
Parameters：
    dataSet — 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无
Modify：
    2018/9/6
"""

def splitDataSet(dataSet,axis,value):
    retDataSet = []                     #创建返回的数据集列表
    for featVec in dataSet:             #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #删除axis列的特征值
            reducedFeatVec.extend(featVec[axis+1:])#将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet

"""
函数说明：选择最优特征
Parameter：
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大特征的索引值
Modify：
    2018/9/6
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #特征个数
    baseEntropy = calcShannonEnt(dataSet)   #原始数据集的经验熵
    bestInfoGain = 0.0  #初始化信息增益
    bestFeature = -1    #最优特征的索引值
    for i in range(numFeatures): #对每个特征计算信息增益
        featList = [example[i] for example in dataSet] #返回某一特征的列向量
        uniqueVals = set(featList)  #创建set集合，其内元素不可重复
        newEntropy = 0.0    #经验条件熵
        for value in uniqueVals:    #计算特征下不同类别的经验条件熵
            subDataSet = splitDataSet(dataSet,i,value) #划分子集
            prob = len(subDataSet)/float(len(dataSet))  
            newEntropy += prob * calcShannonEnt(subDataSet) #某一特征下某一类别的经验条件熵
        infoGain = baseEntropy - newEntropy #信息增益
        if (infoGain > bestInfoGain):   #更新信息增益，找到最大的信息增益
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature

"""
函数说明：统计classlist中出现此处最多的元素
Parameter：
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素
Modify：
    2018/9/6
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items,key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

"""
函数说明：创建决策树
Parameter：
    dataSet - 训练数据集合
    labels - 分类属性标签
    featLabels - 存储选择的最有特征标签
Returns：
    myTree - 决策树
Modify：
    2018/9/6
"""
def createTree(dataSet,labels,featLabels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabels = labels[bestFeat]
    featLabels.append(bestFeatLabels)
    myTree = {bestFeatLabels:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabels][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree

"""
函数说明：使用决策树分类
Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabels - 分类结果
Modify：
    2018/9/6
"""
def classify(inputTree,featLabels,testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabels = classify(secondDict[key],featLabels,testVec)
            else:
                classLabels = secondDict[key]
    return classLabels

if __name__=='__main__':
    dataSet,labels = createDataSet()
    #print(calcShannonEnt(dataSet))
    #print(chooseBestFeatureToSplit(dataSet))
    featLabels = []
    myTree = createTree(dataSet,labels,featLabels)
    print(myTree)
    testVec = [0,1]
    result = classify(myTree,featLabels,testVec)
    if result =='yes':
        print('放贷')
    if result =='no':
        print('不放贷')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    



















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        