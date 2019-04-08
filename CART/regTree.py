# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:35:05 2018

@author: sjdsb
"""

from numpy import *
import matplotlib.pyplot as plt
'''
函数说明：加载数据集
Parameters:
    filename - 包含数据的文件
Returns:
    dataMat - 数据集矩阵
Modify：
    2018/10/26
'''

def loadDataSet(filename):
    dataMat = []            #初始化返回矩阵
    fr = open(filename)     #打开数据文件
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))    #将每行映射为浮点数
        dataMat.append(fltLine)
    return dataMat

"""
    函数说明:绘制数据集
    Parameters:
        filename - 数据集文件
    Returns:
        无
    Modify:
        2018/10/29
"""
def plotDataSet(filename):
    dataMat = loadDataSet(filename)                                        #加载数据集
    n = len(dataMat)                                                    #数据个数
    x = []; y = []                                                #样本点
    for i in range(n):                                                    
        x.append(dataMat[i][1]); y.append(dataMat[i][2])        #样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(x, y, s = 15, c = 'red',alpha = .9)                #绘制样本点
    plt.title('DataSet Visual')                                                #绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

'''
函数说明：将数据集按照某个特征值二分
Parameters:
    dataSet - 数据集
    feature - 要划分的特征
    value - 该特征下的某一个特征值
Returns：
    mat0,mat1 - 划分后的数据矩阵
Modify：
    2018/10/26
'''

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    
    return mat0,mat1



'''
函数说明：生成叶节点，得到目标变量的均值
Parameters:
    dataSet - 数据集
Returns：
    数据集目标变量的均值、
Modify：
    2018/10/27
'''
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

'''
函数说明： 计算目标变量的平方误差
Parameters：
    dataSet - 数据集
Returns：
    数据及目标变量的总方差
Modify：
    2018/10/27
'''

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]


'''
函数说明：寻找数据的最佳二元切分方式
Parameters:
    dataSet - 数据集
    leafType - 均值计算函数
    errType - 总方差计算函数
Returns:
    bestIndex - 最佳划分特征索引值
    bestValue - 最佳划分特征值
Modify：
    2018/10/27
'''

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #tols是容许的误差最小的下降值，tolN是切分的最少样本数
    tolS = ops[0];tolN = ops[1]
    #如果目标变量所有值相等，则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    #数据集的行列数
    m,n = shape(dataSet)
    #计算总方差
    S = errType(dataSet)
    #初始化最优误差，最优特征索引，最优特征值
    bestS = inf; bestIndex = 0;bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            #如果某个子集大小小于tolN，则本次循环不进行切分
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            #子集总误差
            newS = errType(mat0) + errType(mat1)
            #如果误差估计更小，则更新最优特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果切分后的误差下降值很小，则退出
    if (S - bestS) < tolS:
        return None,leafType(dataSet)
    return bestIndex,bestValue
        
'''
函数说明：构建树函数
Parameters:
    dataSet - 数据集
    leafType - 建立叶子结点的函数
    errType - 误差计算函数
    ops - 包含构建树所需其他参数的元组
Returns：
    retTree - 输出树结构
Modify：
    2018/10/26
'''

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #寻找最佳的二元切分方式
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    #满足停止条件返回叶子结点值
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    
    return retTree        
   
'''
函数说明：判断输入变量是否是一颗树
Parameters：
    obj - 测试对象
Returns：
    True or False
Modify:
    2018/10/29
'''
def isTree(obj):
    import types
    return (type(obj).__name__ =='dict')

'''
函数说明：对树进行塌陷处理
Parameters：
    tree - 树
Return:
    树的平均值
Modify：
    2018/10/29
'''
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])    
    return(tree['left'] + tree['right']) / 2.0
    
'''
函数说明：后剪枝处理
Parameters:
    tree - 树
    test - 测试集
Rerurns：
    后剪枝树
Modify：
    2018/10/29
'''
def prune(tree,testData):
    #如果测试集为空，对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)
    #如果有左子树或者右子树，则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    #处理左子树
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    #处理右子树
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    #如果当前结点的左右节点为叶节点的话
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        #计算没有合并的误差
        print(lSet[:,-1])
        print(tree['left'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        #计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        #计算合并的误差
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        #如果合并的误差小于没有合并的误差，那么就合并
        if errorMerge <errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree
'''
函数说明：数据集格式化
Parameters:
    dataSet - 数据集
Returns:
    目标变量X，Y，回归系数ws
Modify:
    2018/10/29
'''
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1];Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('矩阵的逆不存在，不能被转置')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y
'''
函数说明：生成叶子结点
Parameters:
    dataSet - 数据集
Returns:
    回归系数ws
Modify:
    2018/10/29
'''
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

'''
函数说明：计算误差
Parameters:
    dataSet - 数据集
Returns:
    误差
Modify:
    2018/10/29
'''

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))
 


if __name__ == '__main__':
    plotDataSet('ex0.txt')
    #testMat = mat(eye(4))
    #mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    #print(mat0) 
    #plotDataSet('ex0.txt')
    myDat = loadDataSet('exp2.txt')
    myMat = mat(myDat)
    tree = createTree(myMat,modelLeaf,modelErr,(1,10))
    print(tree)
    #x1 = arange(0,0.285477,0.01)
    #y1 = tree['right'][1,0] * x1 + tree['right'][0,0]
    #plt.plot(x1,y1,linewidth=2,color='black')
    #x2 =arange(0.285477,1.0,0.01)
    #y2 = tree['left'][1,0] * x2 + tree['left'][0,0]
    #plt.plot(x2,y2,linewidth=2,color='black')
    #test_filename = 'ex2test.txt'
    #test_Data = loadDataSet(test_filename)
    #test_Mat = mat(test_Data)
    #pruneResult = prune(tree, test_Mat)
    #print(pruneResult)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    









