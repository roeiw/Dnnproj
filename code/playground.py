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
import mssim

#basic script to train models, used in both projects.
model_path = "../models/"
# loaded_model_path = '../models/res_model4.pt'

print(os.getcwd())
print(os.listdir())
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    SIDD_Dataset.rotate_by_90_mul([0,90,180,270]),
    transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

# train_dataset = SIDD_Dataset.SIDD("../patches_128/image_csv1.csv","../patches_128/", transform)
train_dataset = SIDD_Dataset.SIDD("../patches_and_shit24622.csv","./Patches/", transform)


validation_dataset = SIDD_Dataset.SIDD("../validation_and_shit.csv","../val_128/", transform)
#



# im1 = cv2.imread("Patches/0029_001_IP_00800_01000_5500_N/GT_18_3_010.PNG")


train_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
validation_dataloders = torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=True, num_workers=2)

# cv2.imshow("loaders",train_dataloders)
model = ridnet.RIDNET(args)


loss = utility.LabLoss_L1
t = Trainer(args, validation_dataloders,train_dataloders, model, loss, utility.checkpoint(args), model_path+'LabL1_halfrefined_24622.pt')
t.train()
