from multiprocessing.spawn import _main
import numpy as np 
import pandas as pd
import operator
from math import log

from pip import main

def processdata(filepath,data):
    with open(filepath) as f:
        read_data = f.read()
        a = read_data.split()
        m, n = data.shape
        for i in range(0,m):
            for j in range(0,n):
                #print(a[i*181+j*3],a[i*181+j*3+1],a[i*181+j*3+2])
                if  j==n-1:
                    if a[i*181+j*3]=='1;':
                        data[i][j] = 1
                    elif a[i*181+j*3] == '2;':
                        data[i][j] =2
                    elif a[i*181+j*3] =='3;':
                        data[i][j] =3
                    #print(data[i][j])
                elif a[i*181+j*3]=='0' and a[i*181+j*3+1]=='0' and a[i*181+j*3+2]=='0':
                        data[i][j] = 0
                        #print(data[i][j])
                elif a[i*181+j*3]=='0' and a[i*181+j*3+1]=='0' and a[i*181+j*3+2]=='1':
                        data[i][j] = 1

                        #print(data[i][j])
                elif a[i*181+j*3]=='0' and a[i*181+j*3+1]=='1' and a[i*181+j*3+2]=='0':
                        data[i][j] = 2
                    
                        #print(data[i][j])
                elif a[i*181+j*3]=='1' and a[i*181+j*3+1]=='0' and a[i*181+j*3+2]=='0':
                        data[i][j] = 4
                    
                        #print(data[i][j])
                
        print(data)
    f.closed
    return data 





def infor_entropy(data):
    """ 
    信息熵计算
    :param data：数据集
    :return：信息熵
    """
    num = len(data)   
    labelCounts = {}                                #保存每个标签(Label)出现的次数的字典
    for featVec in data:
        currentlabel = featVec[-1]                  #提取每组的标签信息
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0           #如果标签不在字典中，添加进去
        labelCounts[currentlabel] += 1
    entorpy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / num        #选择该标签的概率
        entorpy -= prob * log(prob,2) 
    
    return entorpy

""" 
得到某一特征下其中一种值的数据集（除去该特征值的列）
parameters:
    data - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征其中一个的值
"""
def splitDataSet(data, axis, value):
    retDataSet= []                      #创建返回的数据集
    for featVec in data:                #数组按行读取；dataframe按列读取，此处是数组
        if featVec[axis] == value:
            redecedFeatVec=np.concatenate((featVec[:axis],featVec[axis+1:]),axis=0)      #去除该特征的列,axis=0,按行拼接
            retDataSet.append(redecedFeatVec)
    return retDataSet

def chooseBestFeatureToSplite(data):
    numFeatures = len(data[0]) - 1             #特征数量
    baseEntropy = infor_entropy(data)           #计算数据集的信息熵
    bestInfoGain = 0.0                          #信息增益
    bestFeature = -1                            #信息增益最大的特征索引
    for i in range(numFeatures):
        featlist = [example[i] for example in data]
        uniqueVals = set(featlist)              #创建set()集合，元素不可重复
        newEntropy = 0.0                        #经验条件熵
        for value in uniqueVals:
            subDataSet = splitDataSet(data, i, value)  #获取指定特征的指定值的数据集
            prob = len(subDataSet) / float(len(data))
            newEntropy += prob * infor_entropy(subDataSet)
        infoGain = baseEntropy - newEntropy                   #计算信息增益
        print("第%d个特征值的信息增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i                      #记录信息增益最大的特征索引
    return bestFeature
   

""" 
函数说明：统计给定数据集分类结果中出现最多次的类标签
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现次数最多的类标签
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]  = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse= True)  #items()方法将字典中的每一对键值对组成一个元组；operator.itemgetter()：指定数据的维度
    
    return sortedClassCount[0][0]           #返回classList中出现次数最多的元素
    



""" 
函数说明： 递归构建决策树
Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Return:
    myTree - 决策树
"""
def createTree(data, Labels, featLabels):
    classList = [example[-1] for example in data]    #取分类标签，ei -> 1，ie -> 2 ，n  -> 3
    if classList.count(classList[0]) == len(classList):     #如果类别完全相同则停止继续划分
        return classList[0]
    if len(data[0]) == 1:                      #遍历完所有特征时返回出现次数最多的类标签，剩下这个1是类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplite(data)    #选择最优特征,得到其索引值
    bestFeatLabel = Labels[bestFeat]            #最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}                 #根据最优特征的标签生成树，字典中的每一个键对应的值都是一个字典
    del(Labels[bestFeat])                       #删除已经使用的特征的标签
    featValues = [example[bestFeat] for example in data]    #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                #去掉重复的属性值
    for value in uniqueVals:
        subLabels =  Labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(data, bestFeat,value), subLabels, featLabels)
    return myTree

""" 
函数说明：使用决策树执行分类
Parameters：
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testdata - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))            #获得决策树结点；用iter()将inputTree字典转化为迭代器(iterator)被next()函数不断调用
    secondDict = inputTree[firstStr]            #下一层字典(就是值所在的那个字典)
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
         if testVec[featIndex] == key:
             if type(secondDict[key]) == dict:          #如果值是字典类型的话，说明还要往下划分  !!!!! dict不能加'',找了一晚上，mmp
                return classify(secondDict[key], featLabels, testVec)
             else:
                return secondDict[key]
 
if __name__ == "__main__":

    data = np.zeros((2000,61))
    datatest = np.zeros((1186,61))

    filepath = 'dna_test.txt'
    filepatht = 'dna_train.txt'
    testdata= processdata(filepath,datatest)
    data = processdata(filepatht,data)

    data = np.zeros((2000,61))
    datatest = np.zeros((1186,61))
    filepath = 'dna_test.txt'
    filepatht = 'dna_train.txt'
    testdata= processdata(filepath,datatest)
    data = processdata(filepatht,data)

    labels = ['A{}'.format(i) for i in range(0,60)]

    bestFeature = chooseBestFeatureToSplite(data)

    featLabels = []
    Labels = labels.copy()
    mytree = createTree(data,Labels,featLabels)

    count = 0
    for i in range(0,len(testdata)):
        predict = classify(mytree, labels, testdata[i])
        print(predict,testdata[i][-1])
        if predict == testdata[i][-1]:
            count +=1
    print("准确率{}%".format(count/len(testdata)*100))    

