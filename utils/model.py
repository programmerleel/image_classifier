# -*- coding: utf-8 -*-
# @Time    : 2024/02/29 10:39
# @Author  : LiShiHao
# @FileName: model.py
# @Software: PyCharm


import torch
from torch import nn

from utils.loss import ArcLoss


class ImageClassifierModel(nn.Module):
    def __init__(self, repo, model, feature_dim,num_classes,loss_type,source="github",use_gpu = True):
        super().__init__()
        self.repo = repo
        self.model = model
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.source = source
        self.use_gpu = use_gpu

        # pytorch hub 寻找需要的模型
        self.model = torch.hub.load(self.repo, self.model, pretrained=True,source=self.source)
        self.feature = nn.Linear(1000, feature_dim)
        if loss_type == "ArcLoss":
            self.cls = ArcLoss(self.num_classes,self.feature_dim,self.use_gpu)
        else:
            self.cls = nn.Linear(feature_dim,num_classes)

    def forward(self, x):
        return self.feature(self.model(x)),self.cls(self.feature(self.model(x)))


if __name__ == '__main__':
    model = ImageClassifierModel('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', 512,20)
    x = torch.randn(2, 3, 160, 160)
    result = model.forward(x)
    print(result[0].shape,result[1].shape)
