import os
import cv2
import numpy as np
import torch.utils.data
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import random


class LabelProcessor:
    def __init__(self):
        self.colormap = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        self.cm2lbl = self.encode_label_pix(self.colormap)

    def encode_label_pix(self, colormap):  # data process and load.ipynb: 标签编码，返回哈希表
        # 一维数组，简单理解为RGB三通道全部初始为0
        cm2lbl = np.zeros(256 ** 3)
        # i 为具体类别，即索引0-11
        for i, cm in enumerate(colormap):
            # 根据索引cm21b1[480803]查找对应类别
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
            # 返回哈希表
        return cm2lbl

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')  # 将标签转为数组类型
        # idx 为352*480大小的矩阵
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]  # 取RGB通道对应的像素矩阵，进行哈希映射
        # 返回类别（0-11）的矩阵
        # a=np.array(self.cm2lbl[idx], dtype='int64')
        return np.array(self.cm2lbl[idx], dtype='int64')

    def mask_to_onehot(self, mask, palette):
        """
        Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
        hot encoding vector, C is usually 1 or 3, and K is the number of class.
        """
        semantic_map = []
        for colour in palette:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map

    def onehot_to_mask(self, mask, palette):
        """
        Converts a mask (H, W, K) to (H, W, C)
        """
        x = np.argmax(mask, axis=-1)
        colour_codes = np.array(palette)
        x = np.uint8(colour_codes[x.astype(np.uint8)])
        return x


class IsprsDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def rand_crop(self, data, label, height, weight):
        r = random.randint(0, 88)

        data = data[r:r + height, r:r + weight, :]
        label = label[r:r + height, r:r + weight, :]

        return data, label

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext))[:, :, (2, 1, 0)]
        img, mask = self.rand_crop(img, mask, 512, 512)

        if self.transform is not None:
            # augmented为字典类型
            augmented = self.transform(image=img)
            # 做完数据增强，此时img={ndarray:(512,512,3)}，mask={ndarray:(512,512,2)}
            img = augmented['image']
        else:
            together_transform = Compose([
                transforms.RandomRotate90(),  # 随机旋转90°
                transforms.Flip(),  # 水平和垂直翻转
            ])
            augmented = together_transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            img_transform = Compose([
                transforms.Normalize(),  # 归一化
            ])
            augmented = img_transform(image=img)
            img = augmented['image']

        mask = label_processor.encode_label_img(mask)
        ex_dim_mask = np.expand_dims(mask, axis=2)
        palette = [[0], [1], [2], [3], [4], [5]]
        mask_onehot = label_processor.mask_to_onehot(ex_dim_mask, palette)
        mask_onehot = mask_onehot.transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)
        return img, mask_onehot, {'img_id': img_id}


label_processor = LabelProcessor()
