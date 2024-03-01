# -*- coding: utf-8 -*-
# @Time    : 2024/02/28 17:08
# @Author  : LiShiHao
# @FileName: augmentations.py
# @Software: PyCharm

import albumentations
import cv2
import numpy as np
import torch


class Augmentation():
    def __init__(self, img_size=640, augment=True):
        # self.path = path
        self.image_size = img_size
        self.augment = augment

        self.process = albumentations.Compose([
            albumentations.Blur(blur_limit=12),
            albumentations.HorizontalFlip(),
            albumentations.Transpose(),
            # TODO 实现类似tf中类似sample_distorted_bounding_box的函数
            # albumentations.RandomCrop()
            albumentations.RandomGamma(gamma_limit=(12, 12)),
            albumentations.RandomRotate90(),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        ])

    def load_image(self, path):
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image

    def pad_image(self):
        height, width, _ = self.image.shape
        max_len = max(height, width)
        top = int((max_len - height) / 2)
        bottom = max_len - height - top
        left = int((max_len - width) / 2)
        right = max_len - width - left
        return cv2.cvtColor(
            cv2.copyMakeBorder(self.image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
            cv2.COLOR_BGR2RGB)

    def process_image(self):
        aug_image = self.process(image=self.image_pad)["image"]
        bgr_image = cv2.resize(cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR), (self.image_size, self.image_size))
        return bgr_image

    # TODO 考虑图片 mean 和 std
    def transpose_image(self, image):
        return (torch.from_numpy(image) / 255.0).permute(2, 0, 1)

    def _process_with_augmentation(self):
        image = self.process_image()
        return self.transpose_image(image)

    def _process_without_augmentation(self):
        image = cv2.resize(self.image_pad, (self.image_size, self.image_size))
        return self.transpose_image(image)

    def __call__(self, path):
        self.image = self.load_image(path)
        self.image_pad = self.pad_image()
        if self.augment:
            return self._process_with_augmentation()
        else:
            return self._process_without_augmentation()
