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
from torch.nn.functional import one_hot
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


class ImageClassifierDataset(Dataset):
    def __init__(self, path, img_size=224, augment=True, thread_num=16, allow_formats=ALLOW_FORMATS):

        self.path = path
        self.img_size = img_size
        self.thread_num = thread_num
        self.allow_formats = allow_formats
        self.augment = augment
        self.augmentation = Augmentation(self.img_size, self.augment)

        self.label_names = sorted(
            category for category in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, category)))
        self.label_to_index = dict((category, index) for index, category in enumerate(self.label_names))

        self.tasks = []
        self.pool = ThreadPoolExecutor(thread_num)
        for category_path in (os.path.join(self.path, category_dir) for category_dir in self.label_names):
            self.tasks.append(self.pool.submit(index_subdirectory, category_path, self.label_to_index, allow_formats))
        self.all_file_paths = []
        self.all_labels = []
        for task in tqdm(as_completed(self.tasks), total=len(self.tasks),
                         desc="check files and return files path and name"):
            file_paths, labels = task.result()
            self.all_file_paths = self.all_file_paths + file_paths
            self.all_labels = self.all_labels + labels
        self.pool.shutdown()

    def __len__(self):
        return len(self.all_file_paths)

    def __getitem__(self, index):
        path = self.all_file_paths[index]
        label = self.all_labels[index]
        one_hot_label = one_hot(torch.tensor(label), len(self.label_to_index))
        image = self.augmentation(path)
        return image, one_hot_label


def create_dataloader(path, batch_size=32, workers=8):
    dataset = ImageClassifierDataset(path)
    batch_size = min(batch_size, len(dataset))
    # world_size
    nd = torch.cuda.device_count()
    # num_workers
    # nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = distributed.DistributedSampler(dataset, nd, shuffle=True)
    # TODO 修改 num_worker 出现 TypeError: can‘t pickle _thread.RLock object 问题
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=0, pin_memory=True)
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
    for i, (image, label) in enumerate(dataloader):
        print(image.shape)
        print(label.shape)
        exit()
