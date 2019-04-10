# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:43:58 2018

@author: sjdsb
"""
"""
函数说明:中文文本处理
 
Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
Modify:
    2018/9/12
"""
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt 
import random
import os
import jieba

def TextProcessing(folder_path,test_size=0.2):
    folder_list = os.listdir(folder_path)                        #查看folder_path下的文件
    data_list = []                                                #训练集
    class_list = []
    
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)        #根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)                        #存放子文件夹下的txt文件的列表
        
        j = 1
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding = 'utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw,cut_all = False)
            word_list = list(word_cut)
            
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
    data_class_list = list(zip(data_list,class_list))
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size) + 1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list , train_class_list = zip(*train_list)
    test_data_list,test_class_list = zip(*test_list)
    
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    
    all_words_tuple_list = sorted(all_words_dict.items(),key = lambda f:f[1],reverse = True)
    all_words_list,all_words_nums = zip(*all_words_tuple_list)
    all_words_list = list(all_words_list)
    return all_words_list,train_data_list,test_data_list,train_class_list,test_class_list

"""
函数说明：读取文件里的内容，并去重
Parameters:
    words_file - 文件路径
Returns：
    words_set- 读取的内容的set集合
Modify：
    2018/9/12
"""
def MakeWordSet(words_file):
    words_set = set()
    with open(words_file,'r',encoding = 'utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word)>0:
                words_set.add(word)
    return words_set

"""
函数说明:根据feature_words将文本向量化
 
Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-22
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):                        #出现在特征集中，则置1                                               
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list                #返回结果


"""
函数说明：文本特征选取
Parameter：
    all_words_list - 训练集所有文本文档
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns：
    feature - words - 特征集合
Modify:
    2018/9/12
"""
def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:                            #feature_words的维度为1000
            break  
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

"""
函数说明：新闻分类器
Parameter：
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list- 测试集分类标签
Returens:
    test_accuracy - 分类器精度
Modify：
    2018/9/12
"""
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

if __name__ == '__main__':
    #文本预处理
    folder_path = './SogouC/Sample'                #训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
 
    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordSet(stopwords_file)
 
 
    test_accuracy_list = []
    deleteNs = range(0, 1000, 20)                #0 20 40 60 ... 980
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
 
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
"""
if __name__ == '__main__':
    #文本预处理
    folder_path = './SogouC/Sample'                #训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
 
    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordSet(stopwords_file)
 
 
    test_accuracy_list = []
    feature_words = words_dict(all_words_list, 450, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    ave = lambda c: sum(c) / len(c)
    
    print(ave(test_accuracy_list))
        
        
        
        
        
        