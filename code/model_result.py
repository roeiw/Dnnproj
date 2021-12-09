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

def compare_save_images(gt_image, noisy_image,image_path,models,models_names,image_name,transform):

    noisy_image_write = noisy_image.detach().permute(1, 2, 0).numpy()
    # cv2.imwrite(image_path + 'noisy.png', (noisy_image_write * 255))
    noisy_transformed_image = noisy_image.unsqueeze(0)
    logging.basicConfig(filename = "./../logs/"+image_name+".log", level=logging.INFO)
    gt_image = gt_image.detach().permute(1,2,0).numpy()
    # print("Noisy Image PSNR is: " + str(PSNR(cv2.cvtColor(noisy_transformed_image,cv2.COLOR_BGR2RGB), gt_image)) + " SSIM is: " + str(calc_ssim(cv2.cvtColor(noisy_transformed_image,cv2.COLOR_BGR2RGB),gt_image)))
    for i,model in enumerate(models):
        pred_image1 = pass_though_net(model,noisy_transformed_image)
        # pred_image2 = pass_though_net(model2,noisy_transformed_image)
        print(pred_image1.shape)
        print(type(pred_image1))
        # im = Im/(image_path+models_names[i]+'.png')
        cv2.imwrite(image_path+models_names[i]+'.png',bgr2rgb(pred_image1*255))
        print(type(gt_image))
        print(type(pred_image1))
        # psnr =
        print("psnr for "+ models_names[i]+" image is " + str(PSNR(pred_image1,gt_image)) + " ssim is:" + str(calc_ssim(pred_image1,gt_image)))
        logging.info("psnr for "+ models_names[i]+" image is " + str(PSNR(pred_image1,gt_image))+ " ssim is: " + str(calc_ssim(pred_image1,gt_image)))

    # cv2.imwrite(image_path+'Contentloss_model.png',pred_image2*255)


    # print("psnr for content loss image is" + str(PSNR(pred_image2,gt_image))+ "ssim is:" + str(calc_ssim(pred_image2,gt_image)))
    # cv2.imwrite(image_path+'gt.png',(gt_image*255))
    # print("PSNR is: "+PSNR(gt_image,pred_image))


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
#
# def test_image(gt_path, noisy_path, model_path, transform, show = True, psnr = True, ssim = True):
#     # model load
#     model = ridnet.RIDNET(args)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#
#     # open noisy image
#     gt_image = cv2.imread(gt_path)
#     print(type(gt_image))
#     try :
#         # noisy_image = PIL.Image.open(noisy_path)
#         noisy_image = cv2.imread(noisy_path)
#         print(type(noisy_image))
#         # print("nois")
#         if not gt_path is None:
#             print("inside gt_path loading")
#             print(gt_path)
#             # .numpy()
#
#             # print(type(gt_image)+"how you doiun")
#     except :
#         print("no no no no")
#         # noisy_image = noisy_path
#         gt_image = gt_path
#     state = torch.get_rng_state()
#
#     # cv2.imshow("0",noisy_image)
#     # cv2.waitKey()
#
#     transformed_noisy = transform(noisy_image)
#
#     # print(type(transformed_noisy))
#     torch.set_rng_state(state)
#     if gt_path is None:
#         print("no no no no")
#         gt_image = transform(gt_image).numpy() #.permute(1,2,0)(image recieved is rgb in testset
#
#
#     print("no show")
#     transformed_noisy = transformed_noisy.unsqueeze(0)
#
#     predicted_image = pass_though_net(model,transformed_noisy)
#
#     #at this point this is np
#
#     #print(gt_image.shape)
#     #print(predicted_image.shape)
#
#     # loss = nn.L1Loss()
#     # print(loss(gt_image,predicted_image))
#     psnr_val =0
#     if psnr and not gt_path is None:
#        psnr_val = PSNR(predicted_image, gt_image)
#        print("PSNR is: ",psnr_val)
#     ssim_val = 0
#     if ssim and not gt_path is None:
#         ssim_val = calc_ssim(predicted_image,gt_image)
#         print("SSIM is: ",ssim_val)
#     # pred_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
#
#     # cv2.imshow("predicted image", (pred_rgb* 255).astype('uint8'))
#     # cv2.waitKey(0)
#     # print(pred_rgb)
#     #nois_rgb = cv2.cvtColor(transformed_noisy.squeeze(0).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
#     # predicted_name = noisy_path.split('/')[-2] + '_' + noisy_path.split('/')[-1]# + '_pred.PNG'
#     # cv2.imwrite('PycharmProjects/Data/presentation/pred/' + predicted_name, (pred_rgb* 255))
#
#     if show:
#         nois_rgb = cv2.cvtColor(transformed_noisy.squeeze(0).permute(1,2,0).numpy(),cv2.COLOR_BGR2RGB)
#         pred_rgb = cv2.cvtColor(predicted_image,cv2.COLOR_BGR2RGB)
#         cv2.imshow("noisy image",nois_rgb)
#         cv2.imshow("predicted image", pred_rgb)
#         if gt_path != None:
#             cv2.imshow("gt image", gt_image)
#         # noisy_name = noisy_path.split('/')[-2] + '_' + noisy_path.split('/')[-1]
#         # print(noisy_name)
#         # predicted_name = noisy_name.replace('NOISY','PRD')
#         # gt_name = noisy_name.replace('NOISY','GT')
#         # print(?noisy_name)
#         # cv2.imwrite('../../output_results/'+predicted_name,(predicted_image*255).astype('uint8'))
#         # cv2.imwrite('../../output_results/'+noisy_name,(transformed_noisy.squeeze(0).permute(1,2,0).numpy()*255).astype('uint8'))
#         # # cv2.imwrite('../../output_results/'+gt_name,(gt_image*255).astype('uint8'))
#         # cv2.waitKey(0)
#     # nois_rgb = cv2.cvtColor(transformed_noisy.squeeze(0).permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
#     # pred_rgb = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
#     # predicted_name = noisy_path.split('/')[-2] + '_' + noisy_path.split('/')[-1] +'_pred.PNG'
#     # cv2.imwrite('PycharmProjects/Data/presentation/pred/' + predicted_name, pred_rgb)
# #(predicted_image * 255).astype('uint8')
#     return psnr_val, ssim_val

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

def load_im(folder_path,transform):
    gt_im = Image.open(folder_path+"gt.png")
    noisy_im = Image.open(folder_path+"noisy.png")

    return transform(gt_im),transform(noisy_im)


def bgr2rgb(im):
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return im

# N_mat_path = '../test/ValidationNoisyBlocksSrgb.mat'
# gt_mat_path = '../test/ValidationGtBlocksSrgb.mat'
# # noisy_path = '../test/'
#
#
def main():
    # image = cv2.imread("../test_im/17/noisy.png")
    # inage1 = bgr2rgb(image)
    # cv2.imwrite("../test_im/17/noisy2.png",inage1)
    # return 0
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


    # image = cv2.imread('../test_im/165/gt.png')
    # image1 = bgr2rgb(image)
    # cv2.imwrite('../test_im/165/gt1.png',image1)
    # return 0

    # gt_im = cv2.imread('./Patches/0035_002_GP_00800_00350_3200_N/GT_22_22_010.PNG')
    # gt_im = cv2.imread('../../../Data/SIDD_medium/SSID_medium/SIDD_Medium_Srgb/Data/0165_007_IP_00800_00800_3200_N/0165_GT_SRGB_011.PNG')
    # # cv2.imshow("im",gt_im)
    # # cv2.waitKey()
    # # gt_im = cv2.cvtColor(gt_im,cv2.COLOR_BGR2RGB)
    # noisy_im = cv2.imread('../../../Data/SIDD_medium/SSID_medium/SIDD_Medium_Srgb/Data/0165_007_IP_00800_00800_3200_N/0165_NOISY_SRGB_011.PNG')

    model1 = ridnet.RIDNET(args)
    model1_path = '../models/L1_128p_171021.pt'
    model1.load_state_dict(torch.load(model1_path))
    model1.eval()

    model2 = ridnet.RIDNET(args)
    model2_path = '../models/LabL1_128p_111021.pt'
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()

    model3 = ridnet.RIDNET(args)
    model3_path = '../models/25621_l1.pt'
    model3.load_state_dict(torch.load(model3_path))
    model3.eval()

    model4 = ridnet.RIDNET(args)
    model4_path = '../models/LabLoss_17921_l2.pt'
    model4.load_state_dict(torch.load(model4_path))
    model4.eval()

    model5 = ridnet.RIDNET(args)
    model5_path = '../models/LabLoss_13921_l1.pt'
    model5.load_state_dict(torch.load(model5_path))
    model5.eval()

    model6 = ridnet.RIDNET(args)
    model6_path = '../models/27621_l2.pt'
    model6.load_state_dict(torch.load(model6_path))
    model6.eval()

    model7 = ridnet.RIDNET(args)
    model7_path = '../models/ContentLoss_Lab_27921_l2.pt'
    model7.load_state_dict(torch.load(model7_path))
    model7.eval()

    model8 = ridnet.RIDNET(args)
    model8_path = '../models/Y_L1_241021_final.pt'
    model8.load_state_dict(torch.load(model8_path))
    model8.eval()

    model9 = ridnet.RIDNET(args)
    model9_path = '../models/mssim_61121.pt'
    model9.load_state_dict(torch.load(model9_path))
    model9.eval()




    # models = [model1,model2,model3,model4,model5,model6,model7,model8,model9]
    # models_names = ["L1_128","L1Lab_128","L1","LabL2","LabL1","L2","ContentLossLab","Y_L1","msssim"]


    models = [model3,model5,model8,model9]
    models_names = ["L1","LabL1","Y_L1","msssim"]

    # m1_model = ridnet.RIDNET(args)
    # m1_model_path = '../models/LabL1_128p_191021_final.pt'
    # m1_model.load_state_dict(torch.load(m1_model_path))
    # m1_model.eval()
    #
    # ms_l1_model = ridnet.RIDNET(args)
    # ms_l1_model_path = '../models/msssim_L1_91121_final.pt'
    # ms_l1_model.load_state_dict(torch.load(ms_l1_model_path))
    # ms_l1_model.eval()


    # models = [m1_model,ms_l1_model]
    # models_names = ["L1128_6_epoch","msssim__128p_l1_3_epoch"]
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.RandomCrop(1000)
        # transforms.RandomHorizontalFlip()
        # SIDD_Dataset.rotate_by_90_mul([0, 90, 180, 270])
    ])

    # gt_im,noisy_im = create_cropped_images(gt_im,noisy_im,transform)
    # gt_im,noisy_im = load_im("../test_im/"+"17/",transform)
    # compare_save_images(gt_im,noisy_im,'../test_im/17/',models,models_names,"17",transform)
    #
    gt_im, noisy_im = load_im("../Nam/test_images/" + "167/", transform)
    # print*
    # print("Noisy Image PSNR is: " + str(PSNR(cv2.cvtColor(cv2.UMat(noisy_im),cv2.COLOR_BGR2RGB), gt_im)) + " SSIM is: " + str(calc_ssim(cv2.cvtColor(noisy_im,cv2.COLOR_BGR2RGB),gt_im)))

    compare_save_images(gt_im,noisy_im,"../Nam/test_images/1",models,models_names,"167_91221",transform)


    # gt_im, noisy_im = load_im("../Nam/test_images/" + "167/", transform)
    # compare_save_images(gt_im,noisy_im,"../Nam/test_images/167",models,models_names,"167",transform)
    #
    # gt_im, noisy_im = load_im("../Nam/test_images/" + "315/", transform)
    # compare_save_images(gt_im,noisy_im,"../Nam/test_images/315" ,models,models_names,"315",transform)
    #
    # gt_im, noisy_im = load_im("../Nam/test_images/" + "316/", transform)
    # compare_save_images(gt_im, noisy_im, "../Nam/test_images/316", models, models_names, "316", transform)

    # gt_im, noisy_im = load_im("../test_im/" + "60/", transform)
    # compare_save_images(gt_im, noisy_im, '../test_im/60/', models, models_names, "60", transform)
    #
    # gt_im, noisy_im = load_im("../test_im/" + "165/", transform)
    # compare_save_images(gt_im, noisy_im, '../test_im/165/', models, models_names, "165", transform)
    return 0
    transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.RandomHorizontalFlip()
        # SIDD_Dataset.rotate_by_90_mul([0, 90, 180, 270])
    ])
    test_on_test_set('../test_set/gt/','../test_set/noisy/','../models/res_model1.pt',test_image,transform)
    test_on_test_set()
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