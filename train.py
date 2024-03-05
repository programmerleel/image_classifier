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
from utils.util import EarlyStop

def run(rank,args):

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

    train_dataset = ImageClassifierDataset(args.train_data_root,args.img_size,args.augment)
    train_dataloader = create_dataloader(train_dataset,args.train_batch_size)
    val_dataset = ImageClassifierDataset(args.val_data_root, args.img_size, False)
    val_dataloader = create_dataloader(val_dataset,args.val_batch_size)
    stop = EarlyStop(args.patience,args.save_path)

    model = ImageClassifierModel(args.repo,args.model,args.feature_dim,args.num_classes,args.loss_type,args.source)
    if args.callback:
        model.load_state_dict(torch.load(args.save_path))

    if torch.cuda.is_available():
        model = model.cuda(rank)
        model = parallel.DistributedDataParallel(model,device_ids=[rank])

    # 针对学习率使用 warmup 以及 余弦衰减
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,t_initial=args.epochs,lr_min=args.lr_min,warmup_t=args.warmup_epochs,warmup_lr_init=args.warmup_lr_min)
    criterion = LabelSmoothing(smoothing=args.smoothing)
    center = CenterLoss(num_classes=args.num_classes, feature_dim=args.feature_dim)
    # arc = ArcLoss(num_classes=args.num_classes, feature_dim=args.feature_dim)

    for epoch in range(args.epochs):
        print(f'epoch: {epoch + 1}/{args.epochs}')
        # 开启训练模式
        model.train()
        train(args,rank,train_dataloader,model,optimizer,scheduler,criterion,center,epoch)

        # 开启验证模式
        model.eval()
        eval(args,rank,val_dataloader,model,stop)

def train(args,rank,train_dataloader,model,optimizer,scheduler,criterion,center,epoch):
    train_losses = list()
    train_accs = list()
    for i, (image, label) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        image, label = image.cuda(rank), label.cuda(rank)
        # 推理
        feature, cls = model.forward(image)
        # 计算loss
        loss = 0
        if args.loss_type == "":
            loss = criterion.forward(cls, label)
        elif args.loss_type == "CenterLoss":
            loss = criterion.forward(cls, label) + 1/2*center.forward(feature, label)
        elif args.loss_type == "ArcLoss":
            loss = criterion.forward(cls, label)

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


def eval(args,rank,val_dataloader,model,stop):
    val_accs = list()
    with torch.no_grad():
        for i, (image, label) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            image, label = image.cuda(rank), label.cuda(rank)
            # 推理
            feature, cls = model.forward(image)
            if args.loss_type == "ArcLoss":
                predictions = torch.argmax(cls, dim=1)
            else:
                predictions = torch.argmax(cls, dim=1)
            num_correct = sum(predictions.eq(label))
            running_val_acc = float(num_correct) / float(image.shape[0])
            val_accs.append(running_val_acc)

        val_acc = torch.tensor(val_accs).mean()
        print(f'validation accuracy: {val_acc:.2f}')
        stop(model, val_acc)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_root",type=str,help="训练集根目录",default=r"C:\Users\Administrator\Downloads\archive\car parts 50\train")
    parser.add_argument("--val_data_root",type=str,help="验证集根目录",default=r"C:\Users\Administrator\Downloads\archive\car parts 50\valid")
    parser.add_argument("--img_size",type=int,help="图像尺寸",default=224)
    parser.add_argument("--augment",type=bool,help="是否进行数据增强",default=True)
    parser.add_argument("--train_batch_size", type=int, help="训练批尺寸", default=12)
    parser.add_argument("--val_batch_size", type=int, help="验证批尺寸", default=8)
    parser.add_argument("--repo", type=str, help="pytorch hub模型仓库路径 或 本地模型路径", default="NVIDIA/DeepLearningExamples:torchhub")
    parser.add_argument("--model", type=str, help="pytorch hub模型名称 或 本地模型名称", default="nvidia_efficientnet_b0")
    parser.add_argument("--source", type=str, help="模型来源", default="gitHub")
    parser.add_argument("--callback", type=bool, help="是否断点续训", default=True)
    parser.add_argument("--save_path", type=str, help="模型保存位置", default=r"D:\project\image_classifier\models\efficientnet_b0.pth")
    parser.add_argument("--patience", type=int, help="early stop轮次", default=5)
    parser.add_argument("--num_classes", type=int, help="训练集类别数", default=50)
    parser.add_argument("--feature_dim", type=int, help="特征尺寸", default=512)
    parser.add_argument("--loss_type", type=str, help="损失类型", default="ArcLoss")
    parser.add_argument("--epochs", type=int, help="训练轮次", default=100)
    parser.add_argument("--warmup_epochs", type=int, help="训练轮次", default=5)
    parser.add_argument("--smoothing", type=float, help="标签平滑参数", default=0.1)
    parser.add_argument("--lr_min", type=float, help="标签平滑参数", default=0.005)
    parser.add_argument("--warmup_lr_min", type=float, help="标签平滑参数", default=0.001)
    return parser.parse_args(argv)


if __name__ == '__main__':

    torch.multiprocessing.spawn(run, args=(parse_arguments(sys.argv[1:]),), nprocs=1, join=True)