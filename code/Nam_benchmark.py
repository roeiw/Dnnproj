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




N_mat_path = '../Nam/mat/'
# gt_mat_path = '../test_set/mat/ValidationGtBlocksSrgb.mat'

totensor = transforms.ToTensor()



# n_mat = loadmat(N_mat_path)

# print(n_mat.keys())

# print(n_mat['img_mean'].shape)





# gt_mat = loadmat(gt_mat_path)
# print(gt_mat.keys())
# cv2.imshow("image",gt_mat['ValidationGtBlocksSrgb'][8][8])
# cv2.waitKey(0)



# num_img, num_blocks, _ , _ , _ = n_mat['ValidationNoisyBlocksSrgb'].shape

# for i in range(num_img):
#     cv2.imwrite("../test_images/Noisy_test_im"+str(i)+".png",n_mat['ValidationNoisyBlocksSrgb'][i])
#     cv2.imwrite("../test_images/GT_test_im"+str(i)+".png",gt_mat['ValidationGtBlocksSrgb'][i])
#
# print("Done")


model = ridnet.RIDNET(args)
model_path = '../models/LabL1_128p_191021_final.pt'

model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

total_ssim =0
total_psnr =0
# test_path = '../test/0056_003_N6_03200_04000_5500_N/0056_NOISY_SRGB_010.PNG'
# model_result.test_image(None,test_path,model_path,totensor)

# pred_img = model_result.pass_though_net(model,n_mat['ValidationNoisyBlocksSrgb'][3][5])

# plt.imshow(pred_img)



# cv2.imshow("pred_img", pred_img)
num_mats = 0
for path,subdirs,mats in os.walk(N_mat_path):

    print(mats)
    for mat in mats:
        temp_m = loadmat(os.path.join(path,mat))
        print(temp_m.keys())
        noisy_mat = temp_m['img_noisy']
        gt_mat = temp_m['img_mean']
        H,W,_ = noisy_mat.shape
        print(type(noisy_mat))
        for i in range(0,H,int(H/8)):
            for j in range(0,W,int(W/8)):
                num_mats += 1
                cropped_gt = gt_mat[i:i+int(H/8),j:j+int(W/8),:]
                cropped_noisy = noisy_mat[i:i+int(H/8),j:j+int(W/8),:]
                print("before test image")
                psnr,ssim = model_result.test_image(cropped_gt,cropped_noisy,model_path,totensor,show=False)
                print("after test img")
                total_ssim += ssim
                total_psnr += psnr
print("psnr is: " + str(total_psnr / num_mats))
print("ssim is: " + str(total_ssim / num_mats))





