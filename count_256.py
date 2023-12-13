# -*- coding:utf-8 -*-
# @Time: 2023/12/11 21:06
# @Author: TaoFei
# @FileName: count_256.py
# @Software: PyCharm

import os
from PIL import Image

path = "/home/adminroot/taofei/dataset/COCO2017/train2017"  # 将此路径替换为您要统计的文件夹的路径

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(path, filename)
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 256 or height < 256:
                os.remove(image_path)

large_images = 0
small_images = 0

for filename in os.listdir(path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(path, filename)
        with Image.open(image_path) as img:
            width, height = img.size
            if width >= 256 and height >= 256:
                large_images += 1
            else:
                small_images += 1

print(f"大于256x256的图像数量：{large_images}")
print(f"小于256x256的图像数量：{small_images}")
