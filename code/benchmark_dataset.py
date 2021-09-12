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




N_mat_path = '../test_set/mat/ValidationNoisyBlocksSrgb.mat'
gt_mat_path = '../test_set/mat/ValidationGtBlocksSrgb.mat'

n_mat = loadmat(N_mat_path)
gt_mat = loadmat(gt_mat_path)
# print(gt_mat.keys())
# cv2.imshow("image",gt_mat['ValidationGtBlocksSrgb'][8][8])
# cv2.waitKey(0)

num_img, num_blocks, _ , _ , _ = n_mat['ValidationNoisyBlocksSrgb'].shape

num_samples = num_img*num_blocks

model = ridnet.RIDNET(args)
model_path = '../models/27621_l2.pt'

model.load_state_dict(torch.load(model_path))
model.eval()

total_ssim =0
total_psnr =0
totensor = transforms.ToTensor()
# test_path = '../test/0056_003_N6_03200_04000_5500_N/0056_NOISY_SRGB_010.PNG'
# model_result.test_image(None,test_path,model_path,totensor)

# pred_img = model_result.pass_though_net(model,n_mat['ValidationNoisyBlocksSrgb'][3][5])

# plt.imshow(pred_img)



# cv2.imshow("pred_img", pred_img)

for i in range(num_img):
    for j in range(num_blocks):
        psnr,ssim = model_result.test_image(gt_mat['ValidationGtBlocksSrgb'][i][j],n_mat['ValidationNoisyBlocksSrgb'][i][j],model_path,totensor,show=False)

        total_ssim += ssim
        total_psnr += psnr
        # cv2.imshow("pred_img", pred_img)
        # cv2.imshow("gt image", gt_img)
        # cv2.waitKey(0)
    print("done with pic: ", i)
print("Average PSNR is: ", total_psnr/num_samples)
print("Average SSIM is: ", total_ssim/num_samples)
# cv2.imshow("pred_img", pred_img)




