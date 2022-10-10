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
import os

#running this creates a dataset from validation of SIDD


N_mat_path = '../test_set/mat/ValidationNoisyBlocksSrgb.mat'
gt_mat_path = '../test_set/mat/ValidationGtBlocksSrgb.mat'

n_mat = loadmat(N_mat_path)
gt_mat = loadmat(gt_mat_path)
# print(gt_mat.keys())
# cv2.imshow("image",gt_mat['ValidationGtBlocksSrgb'][8][8])
# cv2.waitKey(0)



num_img, num_blocks, _ , _ , _ = n_mat['ValidationNoisyBlocksSrgb'].shape

for i in range(num_img):
    for j in range(num_blocks):
        im = Image.fromarray(n_mat['ValidationNoisyBlocksSrgb'][i][j])
        im.save("../test_images/1Noisy_test_im"+str(i)+"_"+str(j)+".png")

        im1 = Image.fromarray(gt_mat['ValidationGtBlocksSrgb'][i][j])
        im1.save("../test_images/1Gt_test_im" + str(i) + "_" + str(j) + ".png")
        # cv2.imwrite("../test_images/Noisy_test_im"+str(i)+"_"+str(j)+".png",n_mat['ValidationNoisyBlocksSrgb'][i][j])
        # cv2.imwrite("../test_images/GT_test_im"+str(i)+"_"+str(j)+".png",gt_mat['ValidationGtBlocksSrgb'][i][j])

