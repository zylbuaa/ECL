#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: eval_metrics.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 17:36
'''
import matplotlib.colors
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve
import torch
import os


'''Confusion Matrix'''
class ConfusionMatrix(object):

    def __init__(self,num_classes:int,labels:list):
        self.matrix=np.zeros((num_classes,num_classes)) #初始化混淆矩阵，元素都为0
        self.num_classes=num_classes #类别数量
        self.labels=labels #类别标签
        self.PrecisionofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SensitivityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.SpecificityofEachClass=[0.0 for cols in range(self.num_classes)]
        self.F1_scoreofEachClass = [0.0 for cols in range(self.num_classes)]
        self.acc = 0.0


    def update(self,pred,label):
       if len(pred)>1:
            for p,t in zip(pred, label): #pred为预测结果，labels为真实标签
                self.matrix[int(p),int(t)] += 1 #根据预测结果和真实标签的值统计数量，在混淆矩阵相应的位置+1
       else:
            self.matrix[int(pred),int(label)] += 1

    def summary(self,File):
        #calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i] #混淆矩阵对角线的元素之和，也就是分类正确的数量
        self.acc = sum_TP/np.sum(self.matrix) #总体准确率
        print("the model accuracy is :{:.4f}".format(self.acc))
        File.write("the model accuracy is {:.4f}".format(self.acc)+"\n")

        #precision,recall,specificity
        table=PrettyTable() #创建一个表格
        table.field_names=["","Precision","Sensitivity","Specificity","F1-score"]
        for i in range(self.num_classes):
            TP=self.matrix[i,i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision=round(TP/(TP+FP),4) if TP+FP!=0 else 0.
            Sensitivity=round(TP/(TP+FN),4) if TP+FN!=0 else 0.
            Specificity=round(TN/(TN+FP),4) if TN+FP!=0 else 0.
            F1_score = round((2*Sensitivity*Precision)/(Sensitivity+Precision),4) if (Sensitivity!=0 and Specificity!=0) else 0

            self.PrecisionofEachClass[i]=Precision
            self.SensitivityofEachClass[i]=Sensitivity
            self.SpecificityofEachClass[i]=Specificity
            self.F1_scoreofEachClass[i] = F1_score

            table.add_row([self.labels[i],Precision,Sensitivity,Specificity,F1_score])
        print(table)
        File.write(str(table)+'\n')
        return self.acc

    def get_f1score(self):
        return self.F1_scoreofEachClass


'''ROC AUC'''
def Auc(pro_list,lab_list,classnum,File):
    pro_array = np.array(pro_list)
    #label to onehot
    lab_tensor = torch.tensor(lab_list)
    lab_tensor = lab_tensor.reshape((lab_tensor.shape[0],1))
    lab_onehot = torch.zeros(lab_tensor.shape[0],classnum)
    lab_onehot.scatter_(dim=1, index=lab_tensor, value=1)
    lab_onehot = np.array(lab_onehot)

    table = PrettyTable()  # 创建一个表格
    table.field_names = ["", "auc"]
    roc_auc = []
    for i in range(classnum):
        fpr,tpr,_ = roc_curve(lab_onehot[:,i],pro_array[:,i])
        auc_i = auc(fpr, tpr)
        roc_auc.append(auc_i)
        table.add_row([i,auc_i])
    print(table)
    File.write(str(table) + '\n')
    print("the average auc: {:.4f}".format(np.mean(roc_auc)))
    # return np.mean(roc_auc)