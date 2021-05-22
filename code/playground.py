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
from torchvision import transforms

model_path = '../models/'

transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = SIDD_Dataset.SIDD('../data/patches/image_csv.csv','./data/patches/image_csv.csv', transform)

train_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_dataloders = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

model = ridnet.RIDNET(args)
loss = nn.L1Loss()

t = Trainer(args, test_dataloders,train_dataloders, model, loss, utility.checkpoint(args), model_path+'model1.pt')
t.train()


