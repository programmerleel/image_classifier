# -*- coding: utf-8 -*-
# @Time    : 2024/02/29 10:39
# @Author  : LiShiHao
# @FileName: model.py
# @Software: PyCharm


import torch
from torch import nn


class ImageClassifierModel(nn.Module):
    def __init__(self, repo, model, num_classes):
        super().__init__()
        self.repo = repo
        self.model = model
        self.num_classes = num_classes
        # pytorch hub 寻找需要的模型
        self.model = torch.hub.load(self.repo, self.model, pretrained=True)
        self.output = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.output(self.model(x))


if __name__ == '__main__':
    model = ImageClassifierModel('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', 20)
    x = torch.randn(2, 3, 224, 224)
    result = model.forward(x)
    print(result.shape)
