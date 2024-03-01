# -*- coding: utf-8 -*-
# @Time    : 2024/02/29 16:20
# @Author  : LiShiHao
# @FileName: train.py
# @Software: PyCharm

import argparse
import os

import torch
from torch import optim
from torch import nn
from torch.nn import parallel
from utils.dataloaders import create_dataloader,ImageClassifierDataset
from utils.model import ImageClassifierModel
import sys
import timm.scheduler
from utils.loss import *
from tqdm import tqdm

def run(args,rank):
    train_dataset = ImageClassifierDataset(args.train_data_root,args.img_size,args.augment)
    train_dataloader = create_dataloader(train_dataset,args.train_batch_size)
    val_dataset = ImageClassifierDataset(args.val_data_root, args.img_size, False)
    val_dataloader = create_dataloader(val_dataset,args.val_batch_size)

    model = ImageClassifierModel(args.repo,args.model,args.num_classes,args.feature_dim)

    if torch.cuda.is_available():
        model = model.cuda(rank)
        model = parallel.DistributedDataParallel(model,device_ids=[rank])

    # 针对学习率使用 warmup 以及 余弦衰减
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,t_initial=args.epochs,lr_min=args.lr_min,warmup_t=args.warmup_epochs,warmup_lr_init=args.warmup_lr_min)
    criterion = LabelSmoothing(smoothing=args.smoothing)
    center = CenterLoss(num_classes=args.num_classes, feature_dim=args.feature_dim)
    arc = ArcLoss(num_classes=args.num_classes, feature_dim=args.feature_dim)

    for epoch in range(args.epochs):
        print(f'epoch: {epoch + 1}/{args.epochs}')
        # 开启训练模式
        model.train()
        train(args,rank,optimizer,scheduler,criterion,center,arc,epoch)

        # 开启验证模式
        model.eval()
        eval(rank)

def train(args,rank,optimizer,scheduler,criterion,center,arc,epoch):
    train_losses = list()
    train_accs = list()
    for i, (image, label) in tqdm(enumerate(train_dataloader)):
        image, label = image.cuda(rank), label.cuda(rank)
        # 推理
        feature, cls = model.forward(image)
        # 计算loss
        loss = 0
        if args.loss_type == "":
            loss = criterion.forward(cls, label)
        elif args.loss_type == "CenterLoss":
            loss = criterion.forward(cls, label) + 1/2*center.forward(cls, label)
        elif args.loss_type == "ArcLoss":
            loss = criterion.forward(arc.forward(feature), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        train_losses.append(loss.item())

        # Calculate running train accuracy
        predictions = torch.argmax(cls, dim=1)
        num_correct = sum(predictions.eq(label))
        running_train_acc = float(num_correct) / float(image.shape[0])
        train_accs.append(running_train_acc)

    train_loss = torch.tensor(train_losses).mean()
    train_acc = torch.tensor(train_accs).mean()
    print(f'training loss: {train_loss:.2f}',f'train accuracy: {train_acc:.2f}')


def eval(rank):
    val_accs = list()
    with torch.no_grad():
        for i, (image, label) in tqdm(enumerate(train_dataloader)):
            image, label = image.cuda(rank), label.cuda(rank)
            # 推理
            feature, cls = model.forward(image)


            # Calculate running train accuracy
            predictions = torch.argmax(cls, dim=1)
            num_correct = sum(predictions.eq(label))
            running_train_acc = float(num_correct) / float(image.shape[0])
            val_accs.append(running_train_acc)

        val_acc = torch.tensor(val_accs).mean()
        print(f'validation accuracy: {val_acc:.2f}')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_root",type=str,help="训练集根目录",default=r"C:\Users\Administrator\Downloads\archive\car parts 50\train")
    parser.add_argument("--val_data_root",type=str,help="验证集根目录",default=r"C:\Users\Administrator\Downloads\archive\car parts 50\val")
    parser.add_argument("--img_size",type=int,help="图像尺寸",default=160)
    parser.add_argument("--augment",type=bool,help="是否进行数据增强",default=True)
    parser.add_argument("--train_batch_size", type=int, help="训练批尺寸", default=32)
    parser.add_argument("--val_batch_size", type=int, help="验证批尺寸", default=8)
    parser.add_argument("--repo", type=str, help="pytorch hub模型仓库路径 或 本地模型路径", default="NVIDIA/DeepLearningExamples:torchhub")
    parser.add_argument("--model", type=str, help="pytorch hub模型名称 或 本地模型名称", default=8)
    parser.add_argument("--val_batch_size", type=int, help="验证批尺寸", default=8)
    parser.add_argument("--val_batch_size", type=int, help="验证批尺寸", default=8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    # 设置主进程的 IP 地址和端口号
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    torch.distributed.init_process_group(
        "gloo",
        world_size=-1,
        rank=-1,
    )

    train_dataloader = create_dataloader("C:\collation\category", 16)
    model = ImageClassifierModel('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', len(train_dataloader.dataset.label_to_index))
    x = torch.randn(2, 3, 224, 224)
    result = model.forward(x)
    print(result.shape)
    train(parse_arguments(sys.argv[1:]),0)