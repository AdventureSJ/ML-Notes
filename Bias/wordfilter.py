# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 08:36:06 2018

@author: sjdsb
"""
import numpy as np
"""
函数说明：创建实验样本
Parameter：
    无
Return：
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
Modify：
    2018/9/11
"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

"""
函数说明：根据vocabList词汇表，将inputSet向量化，响亮的每个元素为1或者0
Parameter：
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量，词集模型
Modify：
    2018/9/11
"""
def setofWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for each in inputSet:
        if each in vocabList:
            returnVec[vocabList.index(each)]=1
        else:
            print("the word : %s is not in my Vocabulary"% each)
    return returnVec

"""
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集
Returns：
    vocabSet - 返回不重复的词条列表，也就是词汇表
Modify:
    2018/9/11
"""

def createVocabList(dataSet):
    vocabSet = set([])
    for data in dataSet:
        vocabSet = vocabSet | set(data)
    return list(vocabSet)

"""
函数说明：朴素贝叶斯分类器训练函数
Parameter：
    trainMat — 训练文档矩阵，即setofWord2Vec 返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet 返回的classVec
Returns:
    p0Vect - 非侮辱类的条件概率数组
    p1Vect - 侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
Modify：
2018/9/11
"""

def trainNBO(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)     #文档数目
    numWords = len(trainMatrix[0])      #词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)     #文档属于侮辱类的概率
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0               #初始化概率
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  #向量相加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #print(p1Denom)
    #print(p1Num)
    p1Vect = np.log(p1Num/p1Denom)      #对每个元素做除法
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

"""
函数说明：朴素贝叶斯分类器分类函数
Parameter：
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec - 非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    1 - 属于侮辱类
    0 - 属于非侮辱类
Modify：
    2018/9/14
"""

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    


if __name__ == '__main__':
    postingList,classVec = loadDataSet()
    #print('postingList:\n',postingList)
    myVocabList = createVocabList(postingList)
    #print('myVocabList:\n',myVocabList)
    trainMat=[]
    for postinDoc in postingList:
        trainMat.append(setofWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNBO(np.array(trainMat),np.array(classVec))
    print(p0V)
    print(p1V)
    print(pAb)
    testEntry = ['love', 'my', 'dalmation']									
    thisDoc = np.array(setofWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
         print(testEntry,'属于侮辱类')
    else:
         print(testEntry,'属于非侮辱类')  										
    testEntry = ['stupid', 'garbage']									
    thisDoc = np.array(setofWords2Vec(myVocabList, testEntry))				
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')    
    
    
    
    
    
    
    
    
    
    








































