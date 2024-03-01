# -*- coding: utf-8 -*-
# @Time    : 2024/02/28 15:41
# @Author  : LiShiHao
# @FileName: dataloaders.py
# @Software: PyCharm

from utils.augmentations import Augmentation
from concurrent.futures import as_completed, ThreadPoolExecutor
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, distributed, dataloader
from tqdm import tqdm

ALLOW_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"


# 检查图片后缀是否符合标准
def iter_files(directory, allow_formats):
    for root, _, files in os.walk(directory):
        for file_name in sorted(files):
            if file_name.lower().endswith(allow_formats):
                yield root, file_name


def index_subdirectory(directory, class_indices, allow_formats):
    dir_name = os.path.basename(directory)
    valid_files = iter_files(directory, allow_formats)
    labels = []
    file_paths = []
    for root, file_name in valid_files:
        path = os.path.join(root, file_name)
        if check_file(path):
            labels.append(class_indices[dir_name])
            file_paths.append(path)
    return file_paths, labels


# 避免图片 size 0k 残缺 最快捷的方式
def check_file(path):
    try:
        image = Image.open(path)
        image.load()
        image.close()
        return True
    except:
        return False


def run(path, thread_num, allow_formats):
    label_names = sorted(
        category for category in os.listdir(path) if os.path.isdir(os.path.join(path, category)))
    label_to_index = dict((category, index) for index, category in enumerate(label_names))
    tasks = []
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


class ImageClassifierDataset(Dataset):
    def __init__(self, path, img_size=224, augment=True, thread_num=16, allow_formats=ALLOW_FORMATS):
        self.path = path
        self.img_size = img_size
        self.thread_num = thread_num
        self.allow_formats = allow_formats
        self.augment = augment
        self.augmentation = Augmentation(self.img_size, self.augment)
        self.label_to_index, self.all_file_paths, self.all_labels = run(thread_num, path, allow_formats)

    def __len__(self):
        return len(self.all_file_paths)

    def __getitem__(self, index):
        path = self.all_file_paths[index]
        label = torch.tensor(self.all_labels[index])
        # one_hot_label = one_hot(torch.tensor(label), len(self.label_to_index))
        image = self.augmentation(path)
        # return image, one_hot_label
        return image, label


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

    dataloader = create_dataloader(r"C:\collation\category", batch_size=16)
    for i, data in enumerate(dataloader):
        print(data[0].shape)
        print(data[1].shape)
        exit()
