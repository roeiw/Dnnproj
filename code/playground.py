import os
import option
from data import common
from option import args
from PIL import Image
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

model_path = '../Models2/'
# loaded_model_path = '../models/res_model4.pt'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    SIDD_Dataset.rotate_by_90_mul([0,90,180,270])
    ])
train_dataset = SIDD_Dataset.SIDD("../../../PycharmProjects/Data/SIDD_medium/Patches/image_csv.csv","../../../PycharmProjects/Data/SIDD_medium/Patches/image_csv.csv", transform)
# print(os.getcwd())
# dog_path = '../../../PycharmProjects/Data/presentation/dog_rni15_2.png'
# dog2_path = '../../../PycharmProjects/Data/presentation/pred/presentation_dog_rni15_2.png'

train_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

model = ridnet.RIDNET(args)
# num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
# print("the number of trainable weights is: ", num_trainable_params)
# model.load_state_dict(torch.load(loaded_model_path))
# dog_im = Image.open(dog_path)
# dog_im = transform(dog_im)
# dog2_im = Image.open(dog2_path)
# dog2_im = transform(dog2_im)
# dog2_im = dog2_im.unsqueeze(0)
# dog_im = dog_im.unsqueeze(0)
# dogs = torch.cat((dog_im,dog2_im),dim=0)
# lab_dog = utility.rgb2xyz(dogs)

# print(lab_dog.shape)
#
# print(lab_dog)
# loss = nn.MSELoss()
LabLoss = utility.LabLoss

t = Trainer(args, test_dataloders,train_dataloders, model, LabLoss, utility.checkpoint(args), model_path+'LabLoss_27821_l2.pt')
t.train()


# pic = train_dataset[1000]
# gt = pic["GT_image"]
# noisy = pic["NOISY_image"]
# print(gt.shape)
# print(type(gt))
# cv2.imshow("gt_img", gt.squeeze(0).permute(1,2,0).numpy())
# cv2.imshow("Noisy_img", noisy.squeeze(0).permute(1,2,0).numpy())
# cv2.waitKey(0)
