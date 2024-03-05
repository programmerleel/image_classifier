# -*- coding: utf-8 -*-
# @Time    : 2024/03/05 16:41
# @Author  : LiShiHao
# @FileName: lab.py
# @Software: PyCharm

import os
import shutil

path = r"C:\Users\Administrator\Downloads\archive\car parts 50\test"

for category in os.scandir(path):
    category_name = category.name
    category_path = category.path
    if category_name == "register" or category_name == "recognize":
        continue
    if not os.path.exists(os.path.join(path, "register",category_name)):
        os.makedirs(os.path.join(path, "register",category_name))
    if not os.path.exists(os.path.join(path, "recognize",category_name)):
        os.makedirs(os.path.join(path, "recognize",category_name))
    files = []
    for file in os.scandir(category_path):
        file_name = file.name
        file_path = file.path
        files.append(file_path)
    for i,e in enumerate(files):
        if i < 2:
            shutil.move(e,os.path.join(path, "register",category_name))
        else:
            shutil.move(e, os.path.join(path, "recognize", category_name))
        # pass
    os.rmdir(category_path)