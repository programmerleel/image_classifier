# -*- coding: utf-8 -*-
# @Time    : 2024/03/01 09:26
# @Author  : LiShiHao
# @FileName: test.py
# @Software: PyCharm

import argparse
import os
import sys
import torch
import numpy as np
from utils.augmentation import Augmentation
from utils.database import DBHelper
from utils.model import ImageClassifierModel


def load_model(args):
    model = ImageClassifierModel(args.repo,args.model,args.feature_dim,args.num_classes,args.loss_type,args.source)
    model.load_state_dict(torch.load(args.save_path))

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def image_register(test_data_root,augmentation,model,db_helper):
    for type in os.scandir(test_data_root):
        if type.name != "register":
            continue
        records = []
        for category in os.scandir(type.path):
            category_name = category.name
            category_path = category.path
            # images = []
            # TODO 多张图片同时推理
            center_feature = None
            for file in os.scandir(category_path):
                # file_name = file.name
                file_path = file.path
                if torch.cuda.is_available():
                    image = augmentation(file_path).cuda()
                else:
                    image = augmentation(file_path)
                with torch.no_grad():
                    feature, cls = model.forward(image.unsqueeze(0))
                    # print(feature.shape)
                    # print(feature[0].cpu().numpy())
                if center_feature is None:
                    center_feature = feature[0].cpu().numpy()
                else:
                    center_feature = center_feature + feature[0].cpu().numpy()
                # images.append(image)
            # images = torch.tensor([item.cpu().detach().numpy() for item in images]).cuda()

            center_feature = center_feature / len(os.listdir(category_path))
            records.append((category_name,center_feature))
        db_helper.insert_table_register(records)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_root", type=str, help="测试集根目录", default=r"C:\Users\Administrator\Downloads\archive\car parts 50\test")
    parser.add_argument("--repo", type=str, help="pytorch hub模型仓库路径 或 本地模型路径",
                        default="NVIDIA/DeepLearningExamples:torchhub")
    parser.add_argument("--model", type=str, help="pytorch hub模型名称 或 本地模型名称",
                        default="nvidia_efficientnet_b0")
    parser.add_argument("--loss_type",type=str,help="",default="ArcLoss")
    parser.add_argument("--source", type=str, help="模型来源", default="gitHub")
    parser.add_argument("--save_path", type=str, help="模型保存位置",
                        default=r"D:\project\image_classifier\models\efficientnet_b0.pth")
    parser.add_argument("--num_classes", type=int, help="训练集类别数", default=50)
    parser.add_argument("--feature_dim", type=int, help="特征尺寸", default=512)
    parser.add_argument("--img_size", type=int, help="图像尺寸", default=224)
    parser.add_argument("--augment", type=bool, help="是否进行数据增强", default=False)
    parser.add_argument("--db_path", type=str, help="数据库地址", default=r"C:\Users\Administrator\Downloads\archive\car parts 50\test\results.db")
    return parser.parse_args(argv)

def main(args):
    augmentation = Augmentation(args.img_size,args.augment)
    db_helper = DBHelper(args.db_path)
    model = load_model(args)
    image_register(args.test_data_root,augmentation,model,db_helper)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
