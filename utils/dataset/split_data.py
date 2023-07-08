#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: split_data.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 17:01
'''

import pandas as pd

from utils.project_utils import mkdir_if_not_exist,traverse_dir_files
import os
import shutil
import csv
import random
import argparse

def make_labels_isic(dataset='ISIC2018',label_path=None,ori_path=None,target_path=None):
    '''

    :param dataset: e.g.ISIC2018,ISIC2019
    :param label_path: e.g. os.path.join(datapath,'ISIC_2019_Training_GroundTruth.csv')
    :param ori_path: e.g.os.path.join(datapath, 'ISIC_2019_Training_Input')
    :param target_path: e.g.os.path.join(datapath, 'ISIC2019_Dataset')
    :return:
    '''
    label_df = pd.read_csv(label_path)

    label_dicts = {
        "ISIC2018": {'MEL': 'MEL', 'NV': 'NV', 'BCC': 'BCC', 'AKIEC': 'AK', 'BKL': 'BKL', 'DF': 'DF', 'VASC': 'VASC'},
        "ISIC2019": {'MEL': 'MEL', 'NV': 'NV', 'BCC': 'BCC', 'AK': 'AK', 'BKL': 'BKL', 'DF': 'DF', 'VASC': 'VASC',
                     'SCC': 'SCC'},  # Done
    }
    label_dict = label_dicts[dataset]
    for idx, row in label_df.iterrows():
        name = row['image']
        for label in label_dict.keys():
            if row[label] == 1:
                new_path = os.path.join(target_path, label_dict[label])
                mkdir_if_not_exist(new_path)
                new_img_path = os.path.join(new_path, f"{name}.jpg")
                shutil.copy(os.path.join(ori_path, f"{name}.jpg"), new_img_path)

    print("finish!")

def split_dataset(path,dataset='ISIC2018'):
    '''
    划分数据集
    :return:
    '''
    # image,category,label 3:1:1
    data_path = os.path.join(path,'{}_Dataset'.format(dataset))
    category = os.listdir(data_path) #存储了类别

    #train.csv,val.csv,test.csv
    f_train = open(os.path.join(path,'{}_train.csv'.format(dataset)), 'a+', newline='')
    f_val = open(os.path.join(path,'{}_val.csv'.format(dataset)), 'a+', newline='')
    f_test = open(os.path.join(path,'{}_test.csv'.format(dataset)), 'a+', newline='')

    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    test_writer = csv.writer(f_test)


    headers = ['image','category','label']
    train_writer.writerow(headers)
    val_writer.writerow(headers)
    test_writer.writerow(headers)

    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    test_writer = csv.writer(f_test)

    for label,cate in enumerate(category):
        dir_path = os.path.join(data_path,cate)
        paths_list,_ = traverse_dir_files(dir_path)

        random.shuffle(paths_list)
        for idx,path in enumerate(paths_list):
            row = []
            img_name = path.split('/')[-1].split('.')[0]
            row.append(img_name)
            row.append(cate)
            row.append(label)
            if idx % 5 ==0:
                test_writer.writerow(row)
            elif idx % 5 ==1:
                val_writer.writerow(row)
            else:
                train_writer.writerow(row)
    f_train.close()
    f_val.close()
    f_test.close()
    print("finish!")

parser = argparse.ArgumentParser(description='preprocess the dataset donwloaded from ISIC')
parser.add_argument('--dataset',default='ISIC2018',type=str,help='ISIC2018 or ISIC2019')
parser.add_argument('--datapath',default='/home/ISIC2019',type=str,help='the path of dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    datapath = args.datapath
    label_path = os.path.join(datapath,'ISIC_2019_Training_GroundTruth.csv')
    ori_path = os.path.join(datapath, 'ISIC_2019_Training_Input')
    target_path = os.path.join(datapath, 'ISIC2019_Dataset')
    make_labels_isic(dataset=dataset,label_path=label_path,ori_path=ori_path,target_path=target_path)
    split_dataset(path=datapath,dataset=dataset)
