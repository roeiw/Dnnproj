import os
import torch
import pandas as pd
from skimage import io, transform
from skimage.transform import warp, AffineTransform, rotate
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math

class SIDD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        GT_img_name = str(self.landmarks_frame.iloc[idx, 0])
        GT_img_name = GT_img_name+'.PNG'
        noisy_image_name = GT_img_name.replace("GT", "NOISY")
        GT_image = io.imread(GT_img_name)
        NOISY_image = io.imread(noisy_image_name)
        if self.transform:
            GT_image = self.transform(GT_image)
            NOISY_image = self.transform(NOISY_image)
        sample = {'GT_image': GT_image, 'NOISY_image': NOISY_image}

        return sample
