from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
from skimage import color, feature

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            # img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            img_lr = img.resize((width // 8, height // 8), Image.BICUBIC)
            # 将裁剪过后的图像转换为灰度图
            gray_img_lr = color.rgb2gray(img_lr)
            # 使用canny算子进行边缘检测
            edge_img_lr = feature.canny(gray_img_lr, sigma=1)

            img = np.array(img, dtype=np.uint8)
            img_lr = np.array(img_lr, dtype=np.uint8)
            edge_img_lr = np.array(edge_img_lr, dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, img_lr, edge_img_lr, bb_targets = self.transform((img, img_lr, edge_img_lr, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, img_lr, edge_img_lr, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, img_lrs, edge_img_lrs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        # imgs_lr = imgs
        imgs_lr = torch.stack([resize(img_lr, self.img_size // 8) for img_lr in img_lrs])
        imgs_lr_edge = torch.stack([resize(img_lr_edge, self.img_size // 8) for img_lr_edge in edge_img_lrs])
        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, imgs_lr, imgs_lr_edge, bb_targets

    def __len__(self):
        return len(self.img_files)
