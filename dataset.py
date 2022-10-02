import numpy as np
import cv2
import random

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, edge_paths, mask_paths, edge_mask_paths):
        self.seg_img_paths = img_paths
        self.edge_img_paths = edge_paths
        self.seg_mask_paths = mask_paths
        self.edge_mask_paths = edge_mask_paths

    def __len__(self):
        return len(self.seg_img_paths)

    def __getitem__(self, idx):
        seg_img_path = self.seg_img_paths[idx]
        edge_img_path = self.edge_img_paths[idx]
        seg_mask_path = self.seg_mask_paths[idx]
        edge_mask_path = self.edge_mask_paths[idx]

        image = np.load(seg_img_path)
        edge = np.load(edge_img_path)
        mask = np.load(seg_mask_path)
        edge_mask = np.load(edge_mask_path)

        seg_image = image.transpose((2, 0, 1))
        seg_image = seg_image.astype("float32")
        edge_image = edge.transpose((2, 0, 1))
        edge_image = edge_image.astype("float32")

        WT_Label = mask.copy()
        WT_Label[mask == 1] = 1.
        WT_Label[mask == 2] = 1.
        WT_Label[mask == 4] = 1.
        TC_Label = mask.copy()
        TC_Label[mask == 1] = 1.
        TC_Label[mask == 2] = 0.
        TC_Label[mask == 4] = 1.
        ET_Label = mask.copy()
        ET_Label[mask == 1] = 0.
        ET_Label[mask == 2] = 0.
        ET_Label[mask == 4] = 1.
        seg_label = np.empty((240, 240, 3))
        seg_label[:, :, 0] = WT_Label
        seg_label[:, :, 1] = TC_Label
        seg_label[:, :, 2] = ET_Label
        seg_label = seg_label.transpose((2, 0, 1))

        edge_label = np.empty((240, 240, 1))
        edge_label[:, :, 0] = edge_mask
        edge_label = edge_label.transpose((2, 0, 1))

        seg_label = seg_label.astype("float32")
        edge_label = edge_label.astype("float32")

        return seg_image, edge_image, seg_label, edge_label


