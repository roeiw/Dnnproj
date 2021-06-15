import os
import torch
import torchvision
import pandas as pd
from skimage import io, transform
from skimage.transform import warp, AffineTransform, rotate
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class SIDD(Dataset):

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
            state = torch.get_rng_state()
            GT_image = self.transform(GT_image)
            torch.set_rng_state(state)
            NOISY_image = self.transform(NOISY_image)
        sample = {'GT_image': GT_image, 'NOISY_image': NOISY_image, 'image_name':GT_img_name}

        return sample

class rotate_by_90_mul(object):
    def __init__(self,deg_list):
        self.degree = random.choice(deg_list)

    def __call__(self, image):
        print("before trans",type(image))
        image = TF.rotate(image,self.degree)


        print("after trans",type(image))
        return image








train_dataset = SIDD('../data/patches/image_csv.csv','./data/image_csv.csv', transform=transforms.ToTensor())

train_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

data = next(iter(train_dataloders))
gt = data['GT_image']
nois = data['NOISY_image']
name = data['image_name']
# gt = gt.permute((0,3,1,2))

# print(gt)
# inp = torchvision.utils.make_grid(gt)
#
# nois = nois.permute((0,3,1,2))
# inp2 = torchvision.utils.make_grid(nois)
#
# # print(type(img_list[1]))
#
# print(inp.shape)
#
# imshow(inp)
# plt.show()
# imshow(inp2)
# plt.show()
#
