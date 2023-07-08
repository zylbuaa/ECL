#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: isic.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 17:33
'''

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch

augmentation_rand = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
    transforms.ToTensor()]
    # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
    )

augmentation_sim = transforms.Compose(
    [transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()]
    # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
    )


augmentation_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))
    ])

class isic2019_dataset(Dataset):
    def __init__(self,path,transform,mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_train.csv'))
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_val.csv'))
        else:
            self.df = pd.read_csv(os.path.join(path,'ISIC2019_test.csv'))

    def __getitem__(self, item):
        img_path = os.path.join(self.path,'ISIC2019_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img = Image.open(img_path)
        if (img.mode != 'RGB'):
            img = img.convert("RGB")

        label = int(self.df.iloc[item]['label'])
        label = torch.LongTensor([label])

        if self.transform is not None:
            if self.mode == 'train':
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)

                return [img1,img2],label
            else:
                img1 = self.transform(img)
                return img1, label
        else:
            raise Exception("Transform is None")

    def __len__(self):
        return len(list(self.df['image']))


class isic2018_dataset(Dataset):
    def __init__(self,path,transform,mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_train.csv'))
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_val.csv'))
        else:
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_test.csv'))

    def __getitem__(self, item):
        img_path = os.path.join(self.path,'ISIC2018_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img = Image.open(img_path)
        if (img.mode != 'RGB'):
            img = img.convert("RGB")

        label = int(self.df.iloc[item]['label'])
        label = torch.LongTensor([label])

        if self.transform is not None:
            if self.mode == 'train':
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)

                return [img1,img2],label
            else:
                img1 = self.transform(img)
                return img1, label
        else:
            raise Exception("Transform is None")

    def __len__(self):
        return len(list(self.df['image']))

