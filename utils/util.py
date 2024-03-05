# -*- coding: utf-8 -*-
# @Time    : 2024/03/04 17:32
# @Author  : LiShiHao
# @FileName: util.py
# @Software: PyCharm

import torch

class EarlyStop:
    def __init__(self, patience=1, path='checkpoint.pt'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, model, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
            save_checkpoint(model,self.path)
        elif val_acc <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            save_checkpoint(model,self.path)
            self.counter = 0
        if self.early_stop == True:
            exit(0)

def save_checkpoint(model,path):
    torch.save(model.module.state_dict(), path)
