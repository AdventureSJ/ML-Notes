# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 12:51:39 2018

@author: sjdsb
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import tree
import numpy as np
import pydotplus
from sklearn.externals.six import StringIO

if  __name__ == '__main__':
    with open('lenses.txt','r') as fr:
        lenses = [frp.strip().split('\t') for frp in fr.readlines()]
        lensetarget = []
        for each in lenses:
            target = each[-1]
            lensetarget.append(target)
        #分类的特征
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenseList = []
        lenseDict = {}
        for each in lensesLabels:
            for each1 in lenses:
                lenseList.append(each1[lensesLabels.index(each)])
            lenseDict[each] = lenseList
            lenseList = []
        
        #转换字典数据格式为DataFrame
        lense_pd = pd.DataFrame(lenseDict)
        
        le = LabelEncoder()
        #将每一列标签编码
        for col in lense_pd.columns:
            lense_pd[col] = le.fit_transform(lense_pd[col])
        #设置决策树的深度，创建DecisionTreeClassifier类，然后拟合数据
        clf = tree.DecisionTreeClassifier(max_depth=4)
        clf = clf.fit(lense_pd.values.tolist(),lensetarget)
        #创建一个StringIO()对象
        dot_data = StringIO()
        #使用Graphviz可视化决策树
        tree.export_graphviz(clf,out_file = dot_data,
                             feature_names = lense_pd.keys(),
                             class_names = clf.classes_,
                             filled = True,rounded = True,
                             special_characters = True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("tree.pdf")         #保存树信息
        
        print(clf.predict([[1,1,1,0]]))
        
        