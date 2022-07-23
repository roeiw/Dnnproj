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
import benchmark_dataset
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import random
import model_result
from tqdm import tqdm



def nam_benchmark(model_path):

    N_mat_path = '../../../data/Nam/'
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
    # model_path = '../models/LabL1_syn_15622_fullsynt.pt'

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
    i=0
    num_mats = 0
    for path,subdirs,mats in os.walk(N_mat_path):
        i += 1
        for mat in mats:
            i += 1
            temp_m = loadmat(os.path.join(path,mat))
            # print(temp_m.keys())
            noisy_mat = temp_m['img_noisy']
            gt_mat = temp_m['img_mean']
            H,W,_ = noisy_mat.shape
            factor = 64
            # print(type(noisy_mat))
            # cv2.imwrite("../nam_images/noisy" + str(i) + ".png", cv2.cvtColor(noisy_mat, cv2.COLOR_BGR2RGB))
            # cv2.imwrite("../nam_images/gt"+str(i)+".png",cv2.cvtColor(gt_mat, cv2.COLOR_BGR2RGB))
            for i in tqdm(range(0,H,int(H/factor))):
                for j in range(0,W,int(W/factor)):
                    num_mats += 1
                    cropped_gt = gt_mat[i:i+int(H/factor),j:j+int(W/factor),:]
                    cropped_noisy = noisy_mat[i:i+int(H/factor),j:j+int(W/factor),:]
                    # psnr,ssim = model_result.test_image(cropped_gt,cropped_noisy,model_path,totensor,show=False)

                    psnr,ssim  = model_result.test_image(cropped_gt,cropped_noisy,model_path,totensor,show=False)
                    total_ssim += ssim
                    total_psnr += psnr
    print("psnr is: " + str(total_psnr / num_mats))
    print("ssim is: " + str(total_ssim / num_mats))
    return total_psnr/num_mats, total_ssim/num_mats

def main():
    models = ['../models/LabL1_syn_11522_fullset_and_halfset.pt','../models/LabL1_syn_07522.pt','../models/LabLoss_13921_l1.pt','../models/LabL1_halfrefined_24622.pt','../models/LabL1_syn_10622_thirds.pt','../models/LabL1_syn_15622_fullsynt.pt']
    results = []
    for model in models:
        model_name = model.split(".")[-2].split('/')[-1]
        print(model_name)
        sidd_psnr,sidd_ssim = benchmark_dataset.sidd_benchmark(model)
        nam_psnr,nam_ssim = nam_benchmark(model)
        print("results for model:" + model_name +"is:")
        print("on sidd psnr is: " + str(sidd_psnr) + "ssim is: " + str(sidd_ssim))
        print("on nam psnr is: " + str(nam_psnr) + "ssim is: " + str(nam_ssim))
        result = [sidd_psnr,sidd_ssim,nam_psnr,nam_ssim]
        results.append(result)
    print(results)


if __name__ == '__main__':
   main()




