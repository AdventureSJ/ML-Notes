# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:52:47 2018

@author: sjdsb
"""
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
"""
函数说明：打开并解析文件，对数据进行分类：1代表不喜欢，2代表魅力一般，3代表极具魅力
Parameter：
    filename - 文件名
Return：
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
Modify:
    2018/9/4
"""

def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取每一行，返回列表
    fileLines = fr.readlines()
    #获得行数
    fileRows = len(fileLines)
    #构建返回矩阵
    returnMat = np.zeros((fileRows,3))
    #返回矩阵对应的标签列表
    classLabelVector = []
    index = 0
    for line in fileLines:
        line = line.strip()
        #去除每一行的空白部分以及各种转义字符
        ListFromLine = line.split('\t')
        #将每一行的前三列元素放入返回矩阵中
        returnMat[index,:] = ListFromLine[0:3]
        #将每一行的最后一个元素放入标签列表中
        if ListFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif ListFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif ListFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
            
        index +=1
    return returnMat,classLabelVector


"""
函数说明：对数据进行归一化
Parameters:
    dataSet - 特征矩阵
Return:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
Modify：
    2018/9/4
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges  = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals,(m,1)))/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals



"""
函数说明：数据可视化
Parameters:
    datingDataMat - 特征矩阵
    datingLabels - 分类Label
Return：无
Modify：
    2018/9/4
"""
def showdatas(datingDataMat,datingLabels):
    font = FontProperties(fname = r"c:\windows\fonts\simsun.ttc",size=14)
    fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(13,8))
    
    numberofLabels = len(datingLabels)
    LabelsColors=[]
    for i in datingLabels:
        if i==1:
            LabelsColors.append('black')
        if i==2:
            LabelsColors.append('orange')
        if i==3:
            LabelsColors.append('red')
    
    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=LabelsColors,s=15,alpha=0.5)
    axs0_title_text = axs[0][0].set_title("每年获得的飞行常客里程数与玩视频游戏所消耗的时间占比",
                        FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel("每年获得的飞行常客里程数",FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗的时间',FontProperties=font)
    plt.setp(axs0_title_text,size=9,weight='bold',color='red')
    plt.setp(axs0_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs0_ylabel_text,size=7,weight='bold',color='black')
    
    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=0.5)
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰淇淋公升数',
                        FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'玩视频游戏所消耗的时间',FontProperties=font)
    plt.setp(axs1_title_text,size=9,weight='bold',color='red')
    plt.setp(axs1_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs1_ylabel_text,size=7,weight='bold',color='black')
    
    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=LabelsColors,s=15,alpha=0.5)
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗的时间占比与与每周消费的冰淇淋公升数',
                        FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'玩视频游戏所消耗的时间',FontProperties=font)
    plt.setp(axs2_title_text,size=9,weight='bold',color='red')
    plt.setp(axs2_xlabel_text,size=7,weight='bold',color='black')
    plt.setp(axs2_ylabel_text,size=7,weight='bold',color='black')
    
    didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6,label='didntlike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6,label='smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6,label='largeDoses')
    
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    
    plt.show()
    

    
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
    
    
"""
函数说明：分类器测试函数
Parameters:
    无
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
Modify：
    2018/9/4
"""
def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat,datingLabels = file2matrix(filename)      #数据格式转换
    hoRatio = 0.1       #测试集占样本集的比率
    normMat,ranges,minVals = autoNorm(datingDataMat)        #归一化处理
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)        #测试集数据个数
    errorCount = 0.0
    
    for i in range(numTestVecs):
        classifierResult = classfy1(normMat[i,:],normMat[numTestVecs:m,:],
        datingLabels[numTestVecs:m],3)
        print("分类结果：%d\t真实类别：%d"%(classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率：%f%%"%(errorCount/float(numTestVecs)*100))
    
'''
函数说明：约会网站测试函数
Parameters:
    无
Returns:
    无
Modify:
    2018/10/22
'''

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']  #结果标签
    #输入用户的三个特征
    percentTats = float(input("percentage of time spent playing video games"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    icecream = float(input("liters of ice cream consumed per year"))
    datingDatMat,datingLabels = file2matrix("datingTestSet.txt")
    normMat,ranges,minVals = autoNorm(datingDatMat)
    inArr = np.array([ffMiles,percentTats,icecream])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classfy1(norminArr,normMat,datingLabels,3)
    print("你可能%s这个人" % (resultList[classifierResult-1]))
 

if __name__ =='__main__':
    #filename = "datingTestSet.txt"
    #datingDataMat,datingClassLabel=file2matrix(filename)
    #showdatas(datingDataMat,datingClassLabel)
    #normDataSet,ranges,minVals = autoNorm(datingDataMat)
    #print(normDataSet)
    #print(ranges)
    #print(minVals)
    #datingClassTest()
    classifyPerson()
    
            
            
            
            
        
