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





#
#model.load_state_dict(torch.load(model_path))

#model.eval()
# trans1 = transforms.ToPILImage()
#
#noisy_image = PIL.Image.open(noisy_path)
#gt_image = PIL.Image.open(gt_path)
#state = torch.get_rng_state()
#noisy_image=trans1(noisy_image)
# torch.set_rng_state(state)
# gt_image = transform(gt_image)
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

    #print(gt_image.shape)
    #print(predicted_image.shape)

    # loss = nn.L1Loss()
    # print(loss(gt_image,predicted_image))
    psnr_val =0
    if psnr and not gt_path is None:
       psnr_val = PSNR(predicted_image, gt_image)
       print("PSNR is: ",psnr_val)
    ssim_val = 0
    if ssim and not gt_path is None:
        ssim_val = calc_ssim(predicted_image,gt_image)
        print("SSIM is: ",ssim_val)
    pred_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

    # cv2.imshow("predicted image", (pred_rgb* 255).astype('uint8'))
    # cv2.waitKey(0)
    # print(pred_rgb)
    #nois_rgb = cv2.cvtColor(transformed_noisy.squeeze(0).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    # predicted_name = noisy_path.split('/')[-2] + '_' + noisy_path.split('/')[-1]# + '_pred.PNG'
    # cv2.imwrite('PycharmProjects/Data/presentation/pred/' + predicted_name, (pred_rgb* 255))

    if show:
        nois_rgb = cv2.cvtColor(transformed_noisy.squeeze(0).permute(1,2,0).numpy(),cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(predicted_image,cv2.COLOR_BGR2RGB)
        cv2.imshow("noisy image",nois_rgb)
        cv2.imshow("predicted image", pred_rgb)
        if gt_path != None:
            cv2.imshow("gt image", gt_image)
        # noisy_name = noisy_path.split('/')[-2] + '_' + noisy_path.split('/')[-1]
        # print(noisy_name)
        # predicted_name = noisy_name.replace('NOISY','PRD')
        # gt_name = noisy_name.replace('NOISY','GT')
        # print(?noisy_name)
        # cv2.imwrite('../../output_results/'+predicted_name,(predicted_image*255).astype('uint8'))
        # cv2.imwrite('../../output_results/'+noisy_name,(transformed_noisy.squeeze(0).permute(1,2,0).numpy()*255).astype('uint8'))
        # # cv2.imwrite('../../output_results/'+gt_name,(gt_image*255).astype('uint8'))
        # cv2.waitKey(0)
    # nois_rgb = cv2.cvtColor(transformed_noisy.squeeze(0).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    # pred_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    # predicted_name = noisy_path.split('/')[-2] + '_' + noisy_path.split('/')[-1] +'_pred.PNG'
    # cv2.imwrite('PycharmProjects/Data/presentation/pred/' + predicted_name, pred_rgb)
#(predicted_image * 255).astype('uint8')
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
            path_to_gt_im = path_to_test_gt+'img_'+str(i)+'/block_'+str(j)+'.png'
            path_to_noisy_im = path_to_test_noisy+'img_'+str(i)+'/block_'+str(j)+'.png'
            im_psnr, im_ssim = func(path_to_gt_im, path_to_noisy_im, model_path, transform, show = False, psnr = False, ssim = False)
            print("psnr is: " + str(im_psnr) + "and im ssim is: " + str(im_ssim))
            break
        break





# N_mat_path = '../test/ValidationNoisyBlocksSrgb.mat'
# gt_mat_path = '../test/ValidationGtBlocksSrgb.mat'
# # noisy_path = '../test/'
#
#
def main():
    # print(os.listdir())
    # homepath = '../'
    # return 0
    # nam_path = 'PycharmProjects/Data/Nam/Nam_mat/Nikon_D800/ISO_6400/B_3.mat'
    # nam_mat = loadmat(nam_path)
    # nam_name = (nam_path.split('/')[-2] +'_' + nam_path.split('/')[-1]).replace('mat','PNG')
    # print(nam_name)
    # print(nam_mat.keys())
    # print(nam_mat['img_mean'].shape)
    # print(nam_mat['img_cov'].shape)
    # print(nam_mat['img_noisy'].shape)
    # im1 = nam_mat['img_cov'].astype('uint8')
    # im2 = nam_mat['img_mean'].astype('uint8')
    # print(type(im1))
    # rgb_im = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('PycharmProjects/Data/Nam/Nam_data/D800/'+ nam_name,rgb_im)
    # # cv2.waitKey(0)
    # return 0
    # num_img, num_blocks, _, _, _ = nam_mat['ValidationNoisyBlocksSrgb'].shape

    # make_DB_of_mat('../mats/ValidationNoisyBlocksSrgb.mat','../test_set/noisy/')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        SIDD_Dataset.rotate_by_90_mul([0, 90, 180, 270])
    ])
    test_on_test_set('../test_set/gt/','../test_set/noisy/','../models/LabLoss_6921_l1.pt',test_image,transform)
    # test_image('../test_set/gt/','../test_set/noisy/','../models/LabLoss_6921_l1.pt',transform,show=False,psnr=False,ssim=False)
    return 0
    noisy_path = 'PycharmProjects/Data/presentation/ISO_3200_C_2_cr.png'#'../test/0046_002_G4_00400_00350_3200_L/0046_NOISY_SRGB_010.PNG'
    gt_path = noisy_path.replace('NOISY', 'GT')
    #
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomCrop(850)
         ])
    # #model = ridnet.RIDNET(args)
    model_path = 'PycharmProjects/Dnnproj/models/27621_l2.pt'
    #
    psnr, ssim = test_image(None,noisy_path,model_path,transform, show=False)
    print("psnr is: "+ str(psnr) + " ssim is: "+ str(ssim))
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