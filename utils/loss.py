# -*- coding: utf-8 -*-
# @Time    : 2024/03/01 11:03
# @Author  : LiShiHao
# @FileName: loss.py
# @Software: PyCharm
import torch
from torch import nn

"""
CenterLoss：减少类内距
    （num_classes，feature_dim） 每个类的类中心 使用512维数据表示
    （batch_size，feature_dim） 每张图片的中心 使用512维数据表示
    增加一个 每张图片 到 类中心的距离作为损失
    是的 类内距减小
"""
class CenterLoss(nn.Module):
    def __init__(self,num_classes,feature_dim,use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        # 创建类中心
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes,self.feature_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

    def forward(self,x,labels):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class ArcLoss(nn.Module):
    def __init__(self, num_classes,feature_dim,use_gpu=True,s=32,m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.weights = nn.Parameter(torch.randn(self.num_classes,self.feature_dim).cuda())
        else:
            self.weights = nn.Parameter(torch.randn(self.num_classes, self.feature_dim))

        self.s = s
        self.m = m

    def forward(self, feature):
        #（batch_size,feature_dim）
        x = nn.functional.normalize(feature, dim=1)
        # （num_classes,feature_dim）
        w = nn.functional.normalize(self.weights, dim=1)
        cos = torch.matmul(x, w.t()) / 10
        a = torch.acos(cos)
        top = torch.exp(self.s * torch.cos(a + self.m))  # e^(s * cos(a + m))
        down = torch.sum(torch.exp(self.s * torch.cos(a)), dim=1, keepdim=True) - torch.exp(self.s * torch.cos(a))
        out = torch.log(top / (top + down))
        return out

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
