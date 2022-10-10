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
from math import log10,sqrt
import kornia
import logging
from noise_model.sidd.sidd_utils import kl_div_3_data


def comapre_loaded_im(gt_image, noisy_image,image_path,models,models_names,image_name,transform,just_noisy = False):
    noisy_image_write = transform(noisy_image).permute(1, 2, 0).numpy()
    # print(noisy_image_write)


    noisy_transformed_image = transform(noisy_image).permute(1, 2, 0).numpy()
    logging.basicConfig(filename="./../logs/" + image_name + ".log", level=logging.INFO)
    if just_noisy == False:

        gt_image = transform(gt_image).permute(1, 2, 0).numpy()
        psnr = str(PSNR(gt_image, noisy_transformed_image))
        ssim = str(calc_ssim(gt_image, noisy_transformed_image))
        print(cv2.imwrite(image_path + 'small_gt' + image_name +'.png', gt_image * 255))
        print(cv2.imwrite(image_path + '/' + image_name + "_psnr_is:_"+psnr.replace(".","p") +"_ssim_is:_"+ssim.replace(".","p")+'small_noisy.png', noisy_image_write * 255))
        logging.info("psnr for noisy image is " + psnr + " ssim is: " + ssim)
        print(PSNR(gt_image, noisy_transformed_image))
        print(calc_ssim(gt_image, noisy_transformed_image))
    else :
        gt_image = noisy_transformed_image
    # print("Noisy Image PSNR is: " + str(PSNR(cv2.cvtColor(noisy_transformed_image,cv2.COLOR_BGR2RGB), gt_image)) + " SSIM is: " + str(calc_ssim(cv2.cvtColor(noisy_transformed_image,cv2.COLOR_BGR2RGB),gt_image)))
    for i, load_model in enumerate(models):
        model =load_model_for_eval(load_model)

        pred_image1 = pass_though_net(model, transform(noisy_image).unsqueeze(0))
        # pred_image2 = pass_though_net(model2,noisy_transformed_image)

        # im = Im/(image_path+models_names[i]+'.png')
        psnr = str(PSNR(pred_image1, gt_image))
        ssim = str(calc_ssim(pred_image1, gt_image))
        cv2.imwrite(image_path + image_name + models_names[i] +"_psnr_is:_"+psnr.replace(".","p") +"_ssim_is:_"+ssim.replace(".","p")+'.png',
                    pred_image1*255)

        print("psnr for " + models_names[i] + " image is " + psnr + " ssim is:" + ssim)
        logging.info(
            "psnr for " + models_names[i] + " image is " + psnr + " ssim is: " + ssim)




def compare_save_images(gt_image, noisy_image,image_path,models,models_names,image_name,transform):

    noisy_image_write = noisy_image.permute(1,2,0).numpy()
    # print(noisy_image_write)




    noisy_transformed_image = noisy_image.unsqueeze(0)
    logging.basicConfig(filename = "./../logs/"+image_name+".log", level=logging.INFO)
    gt_image = gt_image.permute(1,2,0).numpy()
    print(cv2.imwrite(image_path + 'small_gt' + image_name+'.png', rgb2bgr(gt_image*255)))
    logging.info("psnr for noisy image is " + str(PSNR(gt_image,noisy_image.permute(1,2,0).numpy())) + " ssim is: " + str(
        calc_ssim(gt_image,noisy_image.permute(1,2,0).numpy())))
    psnr = str(PSNR(gt_image,noisy_image.permute(1,2,0).numpy()))
    ssim = str(calc_ssim(gt_image,noisy_image.permute(1,2,0).numpy()))
    print(cv2.imwrite(image_path + '/' + image_name + '_psnr_is:_'+psnr.replace(".","p") +'_ssim_is:_'+ssim.replace(".","p")+'_small_noisy.png', rgb2bgr(noisy_image_write * 255)))

    # print("Noisy Image PSNR is: " + str(PSNR(cv2.cvtColor(noisy_transformed_image,cv2.COLOR_BGR2RGB), gt_image)) + " SSIM is: " + str(calc_ssim(cv2.cvtColor(noisy_transformed_image,cv2.COLOR_BGR2RGB),gt_image)))
    for i,modelx in enumerate(models):
        model = load_model_for_eval(modelx)

        pred_image1 = pass_though_net(model,noisy_transformed_image)
        # pred_image2 = pass_though_net(model2,noisy_transformed_image)
        psnr = str(PSNR(pred_image1, gt_image))
        ssim = str(calc_ssim(pred_image1, gt_image))
        # im = Im/(image_path+models_names[i]+'.png')
        cv2.imwrite(image_path+image_name+models_names[i]+'_psnr_is:_'+psnr.replace(".","p") +'_ssim_is:_'+ssim.replace(".","p")+'.png',cv2.cvtColor(pred_image1*255, cv2.COLOR_RGB2BGR))

        print("psnr for "+ models_names[i]+" image is " + psnr.replace(".","p") + " ssim is:" + ssim.replace(".","p"))
        logging.info("psnr for "+ models_names[i]+" image is " + psnr.replace(".","p") + " ssim is: " + ssim.replace(".","p"))




def pass_though_net(model, noisy_img):

    pred_img = model(noisy_img)
    return pred_img.squeeze(0).detach().permute(1,2,0).numpy()

def calc_ssim(pred_img,gt_img):

    return ssim(gt_img, pred_img, multichannel=True)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 1000
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def test_image(gt_path, noisy_path, model_path, transform, show = True, psnr = True, ssim = True):
    # model load
    model = ridnet.RIDNET(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # open noisy image
    try :
        noisy_image = PIL.Image.open(noisy_path)
        if not gt_path is None:
            gt_image = PIL.Image.open(gt_path)
    except :
        noisy_image = noisy_path
        gt_image = gt_path
    state = torch.get_rng_state()
    transformed_noisy = transform(noisy_image)

    torch.set_rng_state(state)
    if not gt_path is None:
        gt_image = transform(gt_image).permute(1,2,0).numpy()

    transformed_noisy = transformed_noisy.unsqueeze(0)

    predicted_image = pass_though_net(model,transformed_noisy)

    #at this point this is np



    # loss = nn.L1Loss()
    # print(loss(gt_image,predicted_image))
    psnr_val =0
    if psnr and not gt_path is None:
        psnr_val = PSNR(predicted_image, gt_image)
    ssim_val = 0
    if ssim and not gt_path is None:
        ssim_val = calc_ssim(predicted_image,gt_image)
    if show:
        cv2.imshow("noisy image", transformed_noisy.squeeze(0).permute(1,2,0).numpy())
        cv2.imshow("predicted image", predicted_image)
        if gt_path != None:
            cv2.imshow("gt image", gt_image)
        cv2.waitKey(0)

    return psnr_val, ssim_val


def test_noisy_image(gt_path, noisy_path):
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.RandomCrop(1000)
        # transforms.RandomHorizontalFlip()
        # SIDD_Dataset.rotate_by_90_mul([0, 90, 180, 270])
    ])

    try :
        noisy_image = PIL.Image.open(noisy_path)
        if not gt_path is None:
            gt_image = PIL.Image.open(gt_path)
    except :
        noisy_image = noisy_path
        gt_image = gt_path

    gt = transform(gt_image).permute(1,2,0).numpy()
    noisy = transform(noisy_image).permute(1,2,0).numpy()

    # loss = nn.L1Loss()
    # print(loss(gt_image,predicted_image))

    psnr_val = PSNR(noisy, gt)
    ssim_val = calc_ssim(gt,noisy)

    return psnr_val, ssim_val

def make_DB_of_mat(mat_location,path_to_save):
    mat = loadmat(mat_location)
    mat = mat['ValidationNoisyBlocksSrgb']
    #we get mat of 40 (images),32(blocks),256,256,3
    for i in range(mat.shape[0]):
        img_name = 'img_'+str(i)
        os.mkdir(path_to_save+img_name)
        for j in range(mat.shape[1]):
            #blocks of 256*256*3
            im = mat[i][j]
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            cv2.imwrite(path_to_save+img_name+'/block_'+str(i)+'_'+str(j)+'.png',im)

def test_on_test_set(path_to_test_gt,path_to_test_noisy,model_path,func,transform):
    #num of images
    psnr = 0
    ssim = 0
    count = 0
    for i in range(len(os.listdir(path_to_test_gt))):
        for j in range(len(os.listdir(path_to_test_gt+'img_'+str(i)))):
            j=j+3
            path_to_gt_im = path_to_test_gt+'img_'+str(i)+'/block_'+str(i)+'_'+str(j)+'.png'
            path_to_noisy_im = path_to_test_noisy+'img_'+str(i)+'/block_'+str(i)+'_'+str(j)+'.png'
            im_psnr, im_ssim = func(path_to_gt_im, path_to_noisy_im, model_path, transform, show = False, psnr = True, ssim = True)
            print("psnr is: " + str(im_psnr) + "and im ssim is: " + str(im_ssim))
            break
        break

def create_cropped_images(gt_image,noisy_image,transform):



    state = torch.get_rng_state()
    cropped_GT_image = transform(gt_image)
    torch.set_rng_state(state)
    cropped_NOISY_image = transform(noisy_image)
    return cropped_GT_image,cropped_NOISY_image

def load_im(folder_path,transform,crop):
    gt_im = Image.open(folder_path+"gt.png")
    noisy_im = Image.open(folder_path+"noisy.png")
    if crop:
        cropped_GT_image,cropped_NOISY_image = create_cropped_images(gt_im,noisy_im,transform)
        return  cropped_GT_image,cropped_NOISY_image
    else:
        return transform(gt_im), transform(noisy_im)

def load_nam_im(image_path,img_num,transform,crop):
    gt_im = Image.open(image_path + "gt" + img_num + ".png")
    noisy_im = Image.open(image_path + "noisy" + img_num + ".png")
    if crop:
        cropped_GT_image,cropped_NOISY_image = create_cropped_images(gt_im,noisy_im,transform)
        return  cropped_GT_image,cropped_NOISY_image
    else:
        return transform(gt_im), transform(noisy_im)


def bgr2rgb(im):
    im1 = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im1

def rgb2bgr(im):
    im1 = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im1
# N_mat_path = '../test/ValidationNoisyBlocksSrgb.mat'
# gt_mat_path = '../test/ValidationGtBlocksSrgb.mat'
# # noisy_path = '../test/'
#
#

def test_and_save(noisy_path,model,save_path,transform):
    noisy_im = Image.open(noisy_path)
    model.eval()

    transformed_noisy = transform(noisy_im)
    transformed_noisy = transformed_noisy.unsqueeze(0)

    predicted_image = pass_though_net(model, transformed_noisy)
    cv2.imwrite(save_path, bgr2rgb(predicted_image * 255))

#get images
def get_image_from_mat_db(mat_path):
    mat = loadmat(mat_path) #mat is dict
    print(type(mat['img_mean']))
    print(cv2.imwrite('../Nam/Big_images/gt.png',bgr2rgb(mat['img_mean'])))
    print(cv2.imwrite('../Nam/Big_images/noisy.png',bgr2rgb(mat['img_noisy'])))
    # nam_name = (nam_path.split('/')[-2] +'_' + nam_path.split('/')[-1]).replace('mat','PNG')
    # print(nam_name)
    # print(nam_mat.keys())
    # print(nam_mat['img_mean'].shape)
    # print(nam_mat['img_cov'].shape)
    # print(nam_mat['img_noisy'].shape)
    # im1 = nam_mat['img_cov'].astype('uint8')
    # im2 = nam_mat['img_mean'].astype('uint8')
def get_kld():
    gt_path = "../SNOISY_06_966_0100.png"
    noisy_path = "../SGT_06_966_0100.png"
    gt = cv2.imread(gt_path)
    synt_noisy = cv2.imread(noisy_path)



    _, _, kldiv_fwd2 = kl_div_3_data(gt,noisy)
    print(kldiv_fwd2)

#test funtion for specifc image
def loaded_img_compare_wrp():
    transform = transforms.Compose([
                transforms.ToTensor()
            ])
    base_path = "../../../data/final_present/"
    gt = cv2.imread(base_path+"dog/noisy.png")
    noisy = cv2.imread(base_path+"dog/noisy.png")
    models = ['../models/LabL1_syn_11522_fullset_and_halfset.pt','../models/LabL1_syn_07522.pt','../models/LabLoss_13921_l1.pt','../models/LabL1_halfrefined_24622.pt','../models/LabL1_syn_10622_thirds.pt','../models/LabL1_syn_15622_fullsynt.pt']
    models_names = ["LabL1_syn_11522_fullset_and_halfset","LabL1_syn_07522","LabLoss_13921_l1","LabL1_halfrefined_24622","LabL1_syn_10622_thirds","LabL1_syn_15622_fullsynt"]
    comapre_loaded_im(gt,noisy,base_path+"results/dog/",models,models_names,"dog",transform,just_noisy=True)




#load models and set in evaluation mode
def load_model_for_eval(path):
    model = ridnet.RIDNET(args)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

#get PSNR from image's pathes
def get_PSNR_from_paths(gt_path,noisy_path):
    gt = cv2.imread(gt_path)
    gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
    noisy = cv2.cvtColor(cv2.imread(noisy_path),cv2.COLOR_BGR2RGB)
    cv2.imshow(gt)
    cv2.imshow(noisy)
    cv2.waitKey()
    psnr = PSNR(noisy,  gt)
    ssim = calc_ssim(noisy,gt)
    return psnr,ssim


#from this fuction we send images to be tested on all mentioned model.
#the funtion is per image, and was used for testing of images in both projects.
def main():
    models = ['../models/LabL1_syn_11522_fullset_and_halfset.pt', '../models/LabL1_syn_07522.pt',
              '../models/LabLoss_13921_l1.pt', '../models/LabL1_halfrefined_24622.pt',
              '../models/LabL1_syn_10622_thirds.pt', '../models/LabL1_syn_15622_fullsynt.pt']
    models_names = ["LabL1_syn_11522_fullset_and_halfset", "LabL1_syn_07522", "LabLoss_13921_l1",
                    "LabL1_halfrefined_24622", "LabL1_syn_10622_thirds", "LabL1_syn_15622_fullsynt"]



    # models = [m1_model,ms_l1_model]
    # models_names = ["L1128_6_epoch","msssim__128p_l1_3_epoch"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(1000)])
    #     # transforms.RandomHorizontalFlip()
    #     # SIDD_Dataset.rotate_by_90_mul([0, 90, 180, 270])
    # ])

    #
    gt_im, noisy_im = load_im("../../../data/presnt_im/" + "1822/", transform,True)
    compare_save_images(gt_im, noisy_im, "../../../data/presnt_im/1822/", models, models_names, "1111", transform)


    return 0


if __name__ == '__main__':
    main()
# sig_image = trans1(sig_image)
# print("before net")
#
# cv2.imshow("pred_img",pred_img)
# print("after net")
# plt.savefig('../test/pred_img1.png')
#
#
#
# fig1 = plt.figure(1)
#
#
# plt.imshow(noisy_image.permute(1,2,0))
# plt.savefig('../test/noisy1.png')
#
# fig3 = plt.figure(3)
# plt.imshow(gt_image.permute(1,2,0))
# plt.savefig('../test/gt1.png')
# #print(torch.max(pred_img))
# #print(torch.min(pred_img))
#
# # print(pred_img.shape)
# # print(gt_image.shape)
# # gt_image = transform(gt_image).unsqueeze(0)
#
#
# max_pred = torch.max(gt_image).numpy()
# min_pred = torch.min(gt_image).numpy()
#
# sig_image = pred_img.squeeze(0).detach()
#
#
# # # gt_image = trans1(gt_image)
# # print("pred image shape: ", gt_image.shape)
# # print("sig image shape: ", sig_image.shape)
# # ssim_score = ssim(gt_image.permute(1,2,0).numpy(), sig_image.permute(1,2,0).numpy(), data_range = max_pred-min_pred, multichannel=True)
# # print("ssim score is ", ssim_score)
# # loss = nn.L1Loss()
# # print(loss(gt_image,sig_image))
# print("psnr is: ", utility.calc_psnr(pred_img.detach(),gt_image,1))