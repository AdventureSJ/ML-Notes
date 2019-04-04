# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:46:35 2018

@author: sjdsb
"""

from numpy import *

'''
函数说明：加载数据集
Parameters：
    filename-数据集的文件名
Returns：
    dataMat-返回的数据集
Modify:
    2018/10/15
'''
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():             #遍历数据集每一行                   
        curLine = line.strip().split('\t')  #按制表符分割每一行数据，返回一个列表
        fltLine = list(map(float,curLine))        #将列表中数据类型转换为float类型
        dataMat.append(fltLine)
    return dataMat

'''
函数说明：计算两个向量之间的欧氏距离
Parameters:
    vecA,vecB - 数据点向量
Returns:
    输入向量之间的欧氏距离
Modify：
    2018/10/15
'''
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))    #计算两个向量之间的欧氏距离

'''
函数说明：为给定数据集构建一个包含&个随机质心的集合
Parameters：
    dataSet - 数据集样本
    k - 要分类的类别数
Returns:
    返回簇质心
Modify:
    2018/10/15
'''
def randCent(dataSet,k):
    n = shape(dataSet)[1]                           #数据集的维度，这里坐标是两维
    centroids = mat(zeros((k,n)))                   #初始化质心位置为0
    for j in range(n):
        minJ = min(dataSet[:,j])                    #获取数据集中最小的坐标值
        rangeJ = float(max(dataSet[:,j]) - minJ)    #获取数据集中坐标的范围
        centroids[:,j] = minJ + rangeJ * random.rand(k,1) #随机产生在数据集坐标范围内的质心点
    return centroids

'''
函数说明：k-means聚类算法
Parameters:
    dataSet - 输入样本的矩阵
    k - 质心点也就是要分类的个数
    distMeas - 两个点之间的欧氏距离
    createCent = randCent
Returns:
    centroids - 更新后的质心点坐标
    clusterAssment - 包含着每个数据点对应的簇类别以及和对应簇中心的距离
Modify：
    2018/10/16
'''

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]                   #数据集样本数
    clusterAssment = mat(zeros((m,2)))      
    centroids = createCent(dataSet,k)       #初始化质心点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex =j                   #寻找最近的质心点
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2  #将每个数据点对应的簇（质心类别）以及距离保存在clusterAssment中
        print(centroids)
        for cent in range(k):                           #更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]   #矩阵名.A将mat矩阵转化为array,然后通过数组过滤获得给定簇的所有数据点
            centroids[cent,:] = mean(ptsInClust,axis=0)     #计算所有点的均值
    return centroids,clusterAssment

'''
函数说明:二分k-means聚类算法
Parameters:
    dataSet - 数据集样本
    k - 簇的类别数目
    distMeas - distEclud
Returns:
    mat(centList) - 包含质心数据的矩阵
    clusterAssment - 包含数据点对应质心以及距离平方误差值数据的矩阵
Modify:
    2018/10/17
'''

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]                   #数据集样本数目
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]                   #初始化质心矩阵
    for j in range(m):                      #计算数据集与初始质心之间的SSE
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf                      #初始SSE设为无穷大
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #通过数组过滤获得对应簇i的数据集样本
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) #K -均值算法会生成两个质心(簇），同时给出每个簇的误差值
            sseSplit = sum(splitClustAss[:,1])                                 #当前数据集的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])#未划分数据集的SSE
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE: #更新最小误差值
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #将k-means得到的0、1结果簇重新编号，修改为划分簇及新加簇的编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)  #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]             #将原来的质心点替换为具有更小误差对应的质心点
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss        #重新构建簇矩阵
    return mat(centList), clusterAssment


'''
函数说明：雅虎API地址获取功能封装函数
Parameters:
    stAddress - Portland文件的第一列数据
    city      - Portland文件的第二列数据
Returns:
    json格式的地理编码
Modify：
    2018/10/21
'''
      
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #设置apiStem
    params = {}
    params['flags'] = 'J'    #返回json格式结果
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)   #将创建的字典转换为可以通过URL进行传递的字符串格式
    yahooApi = apiStem + url_params      
    #print (yahooApi)
    c=urllib.urlopen(yahooApi)          #打开URL读取返回值
    return json.loads(c.read())

'''
函数说明：获取地理位置的经纬度信息并添加到原来的地址行最后
Parameters:
    fileName - 存储地理位置的文件
Returns:
    places.txt -输出文件
Modify：
    2018/10/21
'''

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')        #设置输出文件
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])     #获取经纬度信息并添加到原来的那一行后面
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print ("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print ("error fetching")
        sleep(1)        #将函数延迟一面，避免频繁调用导致请求封掉
    fw.close()        
        
#kMeans.massPlaceFind('portlandClub.txt')        

'''
函数说明：球面距离计算函数以及绘图函数
Parameters:
    vecA,vecB - 包含经纬度信息的向量
Returns:
    两个经纬度之间的距离
Modify:
    2018/10/21
'''

def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])          #获取经纬度信息
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    























