# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:22:14 2018

@author: sjdsb
"""

import re
import random
import numpy as np

"""
函数说明：接受一个大字符串并将其解析为字符串列表
Parameter：
    无
Returns:
    无
Modify:
    2018/9/11
"""

def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString)       #将非数字、字母、下划线的字符作为切分标志
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    #除了单个字母，例如大写的I，其余的单词变成小写

"""
函数说明：将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
    dataSet - 整理的样本数据集、
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇列表
Modify：
    2018/9/11
"""
def createVocablist(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


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
函数说明：根据vocabList词汇表，构建词袋模型
Parameter：
    vocabList - creatVocabList
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量
Modify:
    2018/9/11
"""
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

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
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #print(p1Denom)
    #print(p1Num)
    p1Vect = np.log(p1Num/p1Denom)
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
    
"""
函数说明： 测试朴素贝叶斯分类器
Parameters:
    无
Returns:
    无
Modify：
    2018/9/11
"""
def spamTest():
    docList = [];classList = [];fullTest = []
    #导入并解析50个文件，并记录对应的标签值，垃圾邮件标记为1
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt'% i,'r').read())
        docList.append(wordList)
        fullTest.append(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'% i,'r').read())
        docList.append(wordList)
        fullTest.append(wordList)
        classList.append(0)
    #获得50个文件构成的词列表
    vocabList = createVocablist(docList)
    trainingSet = list(range(50));
    testSet = []
    #随机选取测试集合，并从训练集中去除，留存交叉验证
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainClass = []
    #构建训练集词向量列表，训练集标签
    for docIndex in trainingSet:
        trainMat.append(setofWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    #训练算法
    p0V,p1V,pSpam = trainNBO(np.array(trainMat),np.array(trainClass))
    #分类错误个数初始化为0
    errorCount = 0
    #验证算法错误率
    for docIndex in testSet:
        wordVector = setofWords2Vec(vocabList,docList[docIndex]) 
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%'%(float(errorCount)/len(testSet)*100))
    

if __name__ == '__main__':
    spamTest()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    