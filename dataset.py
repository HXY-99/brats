import numpy as np
import cv2
import random

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class BraTS(data.Dataset):

    def __init__(self, root):
        self.root = root
        self.file_list = os.listdir(root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        file = self.file_list[idx]
        img, mask = pkload(os.path.join(self.root, file))

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
        label = np.empty((3, 160, 160))
        label[0, ...] = WT_Label
        label[1, ...] = TC_Label
        label[2, ...] = ET_Label

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        
        return img, label


