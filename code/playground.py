import os
import option
from data import common
from option import args
import PIL
from skimage import io
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from model import ridnet
import SIDD_Dataset
from option import args
from train import Trainer
import torch.nn as nn
import utility
from skimage.metrics import _structural_similarity as ssim
from torchvision import transforms
import cv2

model_path = '../models/'
loaded_model_path = '../models/res_model4.pt'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    SIDD_Dataset.rotate_by_90_mul([0,90,180,270])
    ])
train_dataset = SIDD_Dataset.SIDD('../data/patches/image_csv.csv','./data/patches/image_csv.csv', transform)

train_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
test_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

model = ridnet.RIDNET(args)
num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print("the number of trainable weights is: ", num_trainable_params)
# model.load_state_dict(torch.load(loaded_model_path))
loss = nn.L1Loss()


t = Trainer(args, test_dataloders,train_dataloders, model, loss, utility.checkpoint(args), model_path+'6421_nomean.pt')
#t.train()


pic = train_dataset[1000]
gt = pic["GT_image"]
noisy = pic["NOISY_image"]
print(gt.shape)
print(type(gt))
cv2.imshow("gt_img", gt.squeeze(0).permute(1,2,0).numpy())
cv2.imshow("Noisy_img", noisy.squeeze(0).permute(1,2,0).numpy())
cv2.waitKey(0)
