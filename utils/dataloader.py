# -*- coding: utf-8 -*-
# @Time    : 2024/02/28 15:41
# @Author  : LiShiHao
# @FileName: dataloader.py
# @Software: PyCharm

from utils.augmentation import Augmentation
from concurrent.futures import as_completed, ThreadPoolExecutor
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, distributed, dataloader
from tqdm import tqdm
from dataset import ImageClassifierDataset

ALLOW_FORMATS = "jpeg", "jpg", "png"


# 检查图片后缀是否符合标准 检查图片完整性
def iter_files(directory, allow_formats):
    for root, _, files in os.walk(directory):
        for file_name in sorted(files):
            if file_name.lower().endswith(allow_formats) and check_file(os.path.join(root,file_name)):
                yield root, file_name


def index_subdirectory(directory, class_indices, allow_formats):
    # category directory
    dir_name = os.path.basename(directory)
    # 检查图片
    valid_files = iter_files(directory, allow_formats)
    labels = []
    file_paths = []
    for root, file_name in valid_files:
        path = os.path.join(root, file_name)
        # 标签
        labels.append(class_indices[dir_name])
        # 图片路径
        file_paths.append(path)
    return file_paths, labels


# 检查图片完整性
def check_file(path):
    try:
        image = Image.open(path)
        image.load()
        image.close()
        return True
    except:
        return False


def run(path, thread_num, allow_formats):
    # 标签名
    label_names = sorted(
        category for category in os.listdir(path) if os.path.isdir(os.path.join(path, category)))
    # 对应index
    label_to_index = dict((category, index) for index, category in enumerate(label_names))
    tasks = []
    # 多线程处理
    pool = ThreadPoolExecutor(thread_num)
    for category_path in (os.path.join(path, category_dir) for category_dir in label_names):
        tasks.append(pool.submit(index_subdirectory, category_path, label_to_index, allow_formats))
    all_file_paths = []
    all_labels = []
    for task in tqdm(as_completed(tasks), total=len(tasks),
                     desc="check files and return files path and name"):
        file_paths, labels = task.result()
        all_file_paths = all_file_paths + file_paths
        all_labels = all_labels + labels
    pool.shutdown()
    return label_to_index, all_file_paths, all_labels





def create_dataloader(dataset, batch_size=32, workers=8):
    batch_size = min(batch_size, len(dataset))
    # world_size
    nd = torch.cuda.device_count()
    # num_workers
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = distributed.DistributedSampler(dataset, nd, shuffle=True)
    # TODO 修改 num_worker 出现 TypeError: can‘t pickle _thread.RLock object 问题
    # dataset 里面不能出现多线程！！！
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=nw, persistent_workers=True,
                            pin_memory=True)
    return dataloader


if __name__ == '__main__':
    label_to_index, all_file_paths, all_labels = run(r"C:\0510_1-15_classified\classified",32,ALLOW_FORMATS)
    image_classifier_dataset = ImageClassifierDataset(label_to_index, all_file_paths, all_labels,[160,160],None)
    print(image_classifier_dataset[0][0].shape)
    print(image_classifier_dataset[0][1])
    # # 设置主进程的 IP 地址和端口号
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29501'
    #
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    #
    # torch.distributed.init_process_group(
    #     "gloo",
    #     world_size=-1,
    #     rank=-1,
    # )
    #
    # dataloader = create_dataloader(r"C:\0510_1-15_classified\classified", batch_size=16)
    # for i, data in enumerate(dataloader):
    #
    #     exit()
