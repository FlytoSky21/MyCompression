# -*- coding:utf-8 -*-
# @Time: 2023/11/18 23:16
# @Author: TaoFei
# @FileName: vimeo90k_dataset.py
# @Software: PyCharm


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class Vimeo90KDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        # 随机裁剪为 256x256
        # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 256))
        # img = transforms.functional.crop(img, i, j, h, w)
        img=self.transform(img)

        # 获取裁剪后的 256x256 图像
        img_lr = img.resize((256 // 8, 256 // 8), Image.BICUBIC)

        # 转换为 PyTorch 的 Tensor
        img = transforms.ToTensor()(img)
        img_lr = transforms.ToTensor()(img_lr)

        # # 如果定义了额外的变换，则应用它们
        # if self.transform:
        #     img = self.transform(img)
        #     img_lr = self.transform(img_lr)

        return img, img_lr


if __name__ == "__main__":
    # 示例用法
    data_dir = "/home/adminroot/taofei/dataset/vimeo_septuplet"
    transform = transforms.Compose([transforms.ToTensor()])

    # 创建数据集实例
    vimeo_dataset = Vimeo90KDataset(data_dir, transform=transform)

    # 获取数据集的一个样本
    sample_lr, sample_hr = vimeo_dataset[0]

    plt.imshow(sample_lr)
    plt.show()

    plt.imshow(sample_hr)
    plt.show()

    # 打印样本的形状
    print("Low Resolution Sample Shape:", sample_lr.shape)
    print("High Resolution Sample Shape:", sample_hr.shape)
