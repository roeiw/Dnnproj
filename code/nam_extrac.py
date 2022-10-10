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
import cv2
from skimage import io
from PIL import Image,ImageCms
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import random
import model_result
from tqdm import tqdm


#extract on NAM datset

N_mat_path = '../Nam/mat/'
# total_ssim = 0
# total_psnr = 0

num_mats = 0
for path,subdirs,mats in os.walk(N_mat_path):

    # print(mats)
    for mat in tqdm(mats):
        temp_m = loadmat(os.path.join(path,mat))
        # print(temp_m.keys())
        noisy_mat = temp_m['img_noisy']
        gt_mat = temp_m['img_mean']
        H,W,_ = noisy_mat.shape
        # print(type(noisy_mat))
        for i in range(0,H,int(H/8)):
            for j in range(0,W,int(W/8)):
                num_mats += 1
                cropped_gt = gt_mat[i:i+int(H/8),j:j+int(W/8),:]
                cropped_noisy = noisy_mat[i:i+int(H/8),j:j+int(W/8),:]
                im1 = Image.fromarray(cropped_gt)
                im2 = Image.fromarray(cropped_noisy)
                im1.save("../Nam/images/gt_img_" + str(num_mats) + "_" + str(j) + ".png")
                im2.save("../Nam/images/noisy_img_" + str(num_mats) + "_" + str(j) + ".png")
