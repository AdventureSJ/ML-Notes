# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:34:30 2018

@author: sjdsb
"""

import numpy as np
import operator
"""
函数说明：创建数据集
Parameter：无
Return：
    group - 数据集
    labels - 数据集标签
Modify:
    2018/9/4
"""
def creatdatasets():
    #创建四组数据的二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])      #二维数组第一列表示电影中动作镜头出现的次数，第二列表示亲吻镜头出现的次数。
    #创建数据集的标签
    labels = ['爱情片','爱情片','动作片','动作片']          #电影的标签
    return group,labels
"""
函数说明：KNN算法，分类器
Parameter：
    inx - 测试集
    dataset - 样本集
    labels - 样本集标签
    k - 选择距离最小的K个点
Return：
    sortedClassCount[0][0] - 分类结果
Modify：
    2018/9/4
"""
def classfy1(inx,dataSet,labels,k):
    #获取样本集的行数
    dataSetsize = dataSet.shape[0]
    #将测试集扩充为和样本集一样的结构，并作差值
    diffMat = np.tile(inx,(dataSetsize,1)) - dataSet
    #计算测试集和样本集的距离值
    sqDiffmat = diffMat**2
    sqDisttances = sqDiffmat.sum(axis=1)
    distances = sqDisttances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        #获取前K个元素的类别
        votellabel = labels[sortedDistIndices[i]]
        #计算类别次数
        classCount[votellabel] = classCount.get(votellabel,0) + 1
    #reverse=True降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__=='__main__':
    group,labels = creatdatasets()
    test = [101,20]
    test_class = classfy1(test,group,labels,3)
    print(test_class)