import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self,img_dir,mask_dir):
        self.images = os.listdir(img_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        name = self.images[idx]

        img = cv2.imread(f"{self.img_dir}/{name}")
        img = cv2.resize(img,(256,256))
        img = img.transpose(2,0,1)/255.0

        mask = cv2.imread(f"{self.mask_dir}/{name}",0)
        mask = cv2.resize(mask,(256,256))
        mask = np.expand_dims(mask,0)/255.0

        return torch.tensor(img,dtype=torch.float32), \
               torch.tensor(mask,dtype=torch.float32)