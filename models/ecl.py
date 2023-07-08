#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: ecl.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2023/7/8 15:54
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class balanced_proxies(nn.Module):
    def __init__(self, dim,proxy_num = 3):
        super(balanced_proxies, self).__init__()
        protos = torch.nn.Parameter(torch.empty([proxy_num, dim]))
        self.proxies = torch.nn.init.xavier_uniform_(protos, gain=1)

    def forward(self):
        centers = F.normalize(self.proxies, dim=-1)
        return centers


class ECL_model(nn.Module):
    def __init__(self,num_classes=8,feat_dim=512):
        super(ECL_model, self).__init__()
        cnns = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(*(list(cnns.children())[:-1]))

        self.num_classes = num_classes

        dimension = 512*4

        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.fc = nn.Linear(dimension,self.num_classes)

    def forward(self,x):
        if isinstance(x, list):
            feat1 = self.backbone(x[0])
            feat1 = feat1.view(feat1.shape[0],-1)
            feat1_mlp = F.normalize(self.head(feat1))
            logits = self.fc(feat1)

            feat2 = self.backbone(x[1])
            feat2 = feat2.view(feat2.shape[0], -1)
            feat2_mlp = F.normalize(self.head(feat2))

            return logits,[feat1_mlp,feat2_mlp]

        else:
            feat1 = self.backbone(x)
            feat1 = feat1.view(feat1.shape[0], -1)

            logits = self.fc(feat1)

            return logits