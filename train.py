# -*- coding: utf-8 -*-
# @Time    : 2024/02/29 16:20
# @Author  : LiShiHao
# @FileName: train.py
# @Software: PyCharm

import argparse
import os

import torch
from torch.nn import parallel
from utils.dataloaders import create_dataloader
from utils.model import ImageClassifierModel
import sys


def train(args,rank):

    train_dataloader = create_dataloader(args.train_data_root,args.batch_size)
    model = ImageClassifierModel(args.repo,args.model,len(train_dataloader.dataset.label_to_index))

    if torch.cuda.is_available():
        model = model.cuda(rank)
        model = parallel.DistributedDataParallel(model,device_ids=[rank])



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument()
    return parser.parse_args(argv)

if __name__ == '__main__':
    # 设置主进程的 IP 地址和端口号
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    torch.distributed.init_process_group(
        "nccl",
        world_size=-1,
        rank=-1,
    )

    train_dataloader = create_dataloader("C:\collation\category", 16)
    model = ImageClassifierModel('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', len(train_dataloader.dataset.label_to_index))
    x = torch.randn(2, 3, 224, 224)
    result = model.forward(x)
    print(result.shape)
    train(parse_arguments(sys.argv[1:]))