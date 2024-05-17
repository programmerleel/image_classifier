# -*- coding: utf-8 -*-
# @Time    : 2024/5/16/016 11:15
# @Author  : Shining
# @File    : dataset.py
# @Description : Dataset类
import cv2
import torch
from torch.utils.data import Dataset
from augmentation import Augmentation


class ImageClassifierDataset(Dataset):
    def __init__(self,label_to_index, all_file_paths, all_labels,image_size,augmentation):
        self.label_to_index = label_to_index
        self.all_file_paths = all_file_paths
        self.all_labels = all_labels

        # 数据增强
        self.augmentation = Augmentation(image_size, augmentation)

    def __len__(self):
        return len(self.all_file_paths)

    def __getitem__(self, index):
        path = self.all_file_paths[index]
        label = torch.tensor(self.all_labels[index])
        image = cv2.imread(path)
        image = self.augmentation(path)
        return image, label
