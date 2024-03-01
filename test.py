# -*- coding: utf-8 -*-
# @Time    : 2024/03/01 09:26
# @Author  : LiShiHao
# @FileName: test.py
# @Software: PyCharm


import io
import pickle
import sys
import threading

print(sys.version_info)
lock = threading.RLock()
buffer = io.BytesIO()
pickle.dump(lock, buffer)
