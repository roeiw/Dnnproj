import glob
import logging
import os

import cv2
from scipy.io import savemat

from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
from mylogger import add_logging_level
import sidd.data_loader as loader
import pandas as pd
import numpy as np
import sidd
from sidd.data_loader import check_download_sidd
from sidd.pipeline import process_sidd_image
from sidd.raw_utils import read_metadata, add_noise_to_raw
from sidd.sidd_utils import unpack_raw, kl_div_3_data,process_sidd_image,process_raw_for_save
from statistics import mean,variance
from utility import PSNR
from xl2dict import get_cam_iso_dict
import csv
data_dir = './../../mats/raw_mats/images/'
sidd_path = data_dir #os.path.join(data_dir, 'SIDD_Medium_Raw/Data')
nf_model_path = 'models/NoiseFlow'

samples_dir = os.path.join(data_dir, '../samples')
os.makedirs(samples_dir, exist_ok=True)


def main():
    # Download SIDD_Medium_Raw?
    # check_download_sidd()

    # set up a custom logger
    add_logging_level('TRACE', 100)
    logging.getLogger(__name__).setLevel("TRACE")
    logging.basicConfig(level=logging.TRACE)

    # Prepare NoiseFlow
    # Issue: Low-probability sampling leading to synthesized pixels with too-high noise variance.
    # Solution: Contracting the sampling distribution by using sampling temperature less than 1.0 (e.g., 0.6).
    # Reference: Parmar, Niki, et al. "Image Transformer." ICML. 2018.
    # noise_flow = NoiseFlowWrapper(nf_model_path, sampling_temperature=0.6)

    # sample noise and add it to clean images
    patch_size = 80
    batch_size = 1  # using batches is faster
    # kldiv_list = []
    cam_iso_dict = get_cam_iso_dict()
    # gauss_dict = {}
    # read_dict={}
    # save_num = 5
    with open("../..//syn_set/syn_csv.csv",'w') as csv_file:
        csv_writer = csv.writer(csv_file,lineterminator="\n")
        for sc_id in range(1,200):  # scene IDs
            im_path = glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))
            if (not im_path): continue
            # print(im_path[0])
            # load images
            metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
                glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*METADATA_RAW_010.MAT'))[0])
            # print(cam)
            # print(iso)
            # print(cam_iso_dict.keys())
            lambda_read , lambda_shot, _= cam_iso_dict[str(cam)+"_"+str(iso)]
            lambda_read = float(lambda_read)
            lambda_shot = float(lambda_shot)
            noisy = loader.load_raw_image_packed(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*NOISY_RAW_010.MAT'))[0])

            clean = loader.load_raw_image(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))[0])
            # camera_dict, gauss_dict,read_dict= sidd.sidd_utils.rate_lamdas(cleanraw,noisy,cam,bayer_2by2, wb, cst2,sc_id,camera_dict,iso,gauss_dict,read_dict,save_num)
            # save_num -= 1
            # continue
            #
            # place = str(iso)+str(cam)
            # camera_dict[place]=True



            if iso not in [100, 400, 800, 1600, 3200]:
                continue

            np.random.seed(12345)  # for reproducibility
            n_pat = 1250
            # continue

            read_noise = add_noise_to_raw(clean, lambda_read, lambda_shot)
            read_noise= loader.get_raw_packed(read_noise)
            clean= loader.get_raw_packed(clean)
            new_dir = "../../syn_set/" + str(sc_id) + "/"
            os.mkdir(new_dir)
            for p in range(n_pat):

                # crop patches
                v = np.random.randint(0, clean.shape[1] - patch_size)
                u = np.random.randint(0, clean.shape[2] - patch_size)

                clean_patch,image_path_for_csv = process_raw_for_save(clean,u,v,patch_size,bayer_2by2, wb, cst2,new_dir,sc_id, p, iso,"SGT")
                csv_writer.writerow([image_path_for_csv])
                # noisy_patch = process_raw_for_save(noisy,u,v,patch_size,bayer_2by2, wb, cst2,samples_dir,sc_id, p, iso,"noisy")
                # syn_noise_patch = process_raw_for_save(syn_noisy,u,v,patch_size,bayer_2by2, wb, cst2,samples_dir,sc_id, p, iso,"syn_noisy")
                # gauss_patch = process_raw_for_save(gauss,u,v,patch_size,bayer_2by2, wb, cst2,samples_dir,sc_id, p, iso,"gauss")
                read_noise_patch ,_= process_raw_for_save(read_noise,u,v,patch_size,bayer_2by2, wb, cst2,new_dir,sc_id, p, iso,"SNOISY")
                continue

                clean_patch = np.expand_dims(clean_patch, 0)



                # sample noise
                # noise_patch_syn = noise_flow.sample_noise_nf(clean_patch, 0.0, 0.0, iso, cam)
                # noise_patch_syn = np.squeeze(noise_patch_syn)[1:-1, 1:-1, :]

                # save as .mat
                # save_mat_fn = os.path.join(samples_dir, '%02d_%02d_%04d.mat' % (sc_id, p, iso))
                # savemat(save_mat_fn, {'clean': clean_patch, 'noisy': noisy_patch, 'noisy_syn': noisy_patch_syn,
                #                       'metadata': metadata})

                # compute KL divergence
        #         kldiv_fwd, _, _ = kl_div_3_data(noisy_patch.flatten() - clean_patch.flatten(), syn_noise_patch.flatten() - clean_patch.flatten())
        #         kldiv_fwd1, _, _ = kl_div_3_data(noisy_patch.flatten() - clean_patch.flatten(),
        #                                         gauss_patch.flatten() - clean_patch.flatten())
        #         kldiv_fwd2, _, _ = kl_div_3_data(noisy_patch.flatten() - clean_patch.flatten(),
        #                                          read_noise_patch.flatten() - clean_patch.flatten())
        #         # print("PSNR for noisy: " + str(PSNR(noisy_patch,clean_patch)))
        #         # print("PSNR for syn_noisy: " + str(PSNR(noisy_patch_syn,clean_patch)))
        #         # print("PSNR for gauss_syn_noisy: " + str(PSNR(noisy_patch_syn,clean_patch)))
        #         print("kl_divergance: per image: " + str(kldiv_fwd))
        #         print("kl_divergance1: per image: " + str(kldiv_fwd1))
        #         print("kl_divergance2 read noise per image: " + str(kldiv_fwd2))
        #         kldiv_list.append(kldiv_fwd)
        # for key,value in camera_dict.items():
        #     mean1 = mean(value)
        #     variance1 = variance(value)
        #     print("avarage of shot_noise " + str(key) + " is " + str(mean1) + "_variance is: " + str(variance1))
        # for key, value in gauss_dict.items():
        #     mean2 = mean(value)
        #     variance2 = variance(value)
        #     print("avarage of gauss_noise " + str(key) + " is " + str(mean2) + "_variance is: " + str(variance2))
        # for key, value in read_dict.items():
        #     mean3 = mean(value)
        #     variance3 = variance(value)
        #     print("avarage of read_noise " + str(key) + " is " + str(mean3) + "_variance is: " + str(variance3))
            # camera_dict[key].a
        # print(camera_dict)
        # print("Mean KL divergence = {}".format(np.mean(np.array(kldiv_list), axis=0)))


def load_cam_iso_nlf():
    cin = pd.read_csv('cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


if __name__ == '__main__':
    main()
