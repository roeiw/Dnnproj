import glob
import logging
import os
from tqdm import tqdm
import cv2
from scipy.io import savemat

# from borealisflows.NoiseFlowWrapper import NoiseFlowWrapper
from mylogger import add_logging_level
import sidd.data_loader as loader
import pandas as pd
import numpy as np
import sidd
from sidd.data_loader import check_download_sidd
from sidd.pipeline import process_sidd_image
from sidd.raw_utils import read_metadata, add_noise_to_raw
from sidd.sidd_utils import unpack_raw,  kl_div_3_data,process_sidd_image,process_raw_for_save,get_best_lambdas,generate_noisy_examples
from statistics import mean,variance
# from utility import PSNR
from xl2dict import get_cam_iso_dict, create_lambda_csv,write_csv2
import csv
from utility import PSNR,calc_ssim



data_dir = '../../../../data/raw/SIDD_Medium_Raw/Data/'
sidd_path = data_dir #os.path.join(data_dir, 'SIDD_Medium_Raw/Data')
nf_model_path = 'models/NoiseFlow'

samples_dir = os.path.join(data_dir, '../examples/')
os.makedirs(samples_dir, exist_ok=True)


#create synthetic dataset for a csv with relevant images
def create_synt(csv_path):
    # sample noise and add it to clean images
    patch_size = 42
    with open(csv_path,"w") as csv_file:
        csv_writer = csv.writer(csv_file)
        for sc_id in tqdm(range(1, 200)):  # scene IDs
            im_path = glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))
            if (not im_path): continue
            # print(im_path[0])
            # load images
            metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
                glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*METADATA_RAW_010.MAT'))[0])
            dict = get_cam_iso_dict("../best_lambdas_7622.csv")
            if (str(cam) + "_"+str(iso)) not in dict.keys(): continue

            _ ,lambda_shot , lambda_read= dict[str(cam)+"_"+str(iso)]
            lambda_read = float(lambda_read)
            lambda_shot = float(lambda_shot)
            noisy = loader.load_raw_image_packed(
                glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*NOISY_RAW_010.MAT'))[0])

            clean = loader.load_raw_image(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))[0])
            np.random.seed(12345)  # for reproducibility
            n_pat = 4500
            # # continue
            #
            read_noise, _ , _ = add_noise_to_raw(clean, lambda_read, lambda_shot)
            read_noise= loader.get_raw_packed(read_noise)
            clean= loader.get_raw_packed(clean)
            new_dir = "../../full_synt_set/" + str(sc_id) + "/"
            os.mkdir(new_dir)
            for p in range(n_pat):

                # crop patches
                v = np.random.randint(0, clean.shape[1] - patch_size)
                u = np.random.randint(0, clean.shape[2] - patch_size)

                clean_patch, gt_path= process_raw_for_save(clean,u,v,patch_size,bayer_2by2, wb, cst2,new_dir,sc_id, p, iso,"SGT")
                read_noise_patch,_ = process_raw_for_save(read_noise,u,v,patch_size,bayer_2by2, wb, cst2,new_dir,sc_id, p, iso,"SNOISY")


                gt_path = gt_path.replace("../../","../")
                csv_writer.writerow([gt_path])

#basic fucntion that creates csv with relevant images and the lamdas that are relevant to them.
#this function is based on a function in noise flow work, for working with RAW images.
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


    read_dict={}
    camera_dict={}
    save_num = 5
    with open("../gammas_and_shit2.csv",'w') as csv_file:
        csv_writer = csv.writer(csv_file,lineterminator="\n")
        for sc_id in tqdm([77,130,150,51]):  # scene IDs
            im_path = glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))
            if (not im_path): continue
            # print(im_path[0])
            # load images
            metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
                glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*METADATA_RAW_010.MAT'))[0])

            noisy = loader.load_raw_image_packed(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*NOISY_RAW_010.MAT'))[0])

            clean = loader.load_raw_image(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))[0])
            generate_noisy_examples(clean,noisy,cam,bayer_2by2,wb,cst2,sc_id,iso,"../../../../data/raw/SIDD_Medium_Raw/real_examples_small/")
            continue
            read_dict = sidd.sidd_utils.rate_lamdas(clean,noisy,cam,bayer_2by2, wb, cst2,sc_id,camera_dict,iso,gauss_dict,read_dict,save_num)


        create_lambda_csv(read_dict)



def save_noise_examples_patches():

    for sc_id in tqdm(range(1, 200)):  # scene IDs
        im_path = glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))
        if (not im_path): continue
        metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
                glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*METADATA_RAW_010.MAT'))[0])
        noisy = loader.load_raw_image_packed(
                glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*NOISY_RAW_010.MAT'))[0])

        clean = loader.load_raw_image(glob.glob(os.path.join(sidd_path, '%04d_*' % sc_id, '*GT_RAW_010.MAT'))[0])
        sidd.sidd_utils.generate_noisy_examples(clean, noisy, cam, bayer_2by2, wb, cst2, sc_id, iso)

def compare_pipelines(raw_noisy_path,srgb_noisy_path,metadata,gt_path):
    raw_noisy = loader.load_raw_image(
        raw_noisy_path)
    raw_noisy= loader.get_raw_packed(raw_noisy)
    print(raw_noisy.shape)

    metadata, bayer_2by2, wb, cst2, iso, cam = read_metadata(
        metadata)
    patch_size = 400
    c = 0

    srgb_processed,path = process_raw_for_save(raw_noisy,c,c,patch_size,bayer_2by2, wb, cst2,os.path.abspath("../../../../data/presnt_im/1722/"),5, 0, iso,"roei_hagever",save = True)
    noisy_srgb = cv2.imread(srgb_noisy_path)
    srgb_processed = cv2.imread(path)
    # srgb_processed = cv2.cvtColor(srgb_processed, cv2.COLOR_BGR2RGB)
    srgb_processed = cv2.rotate(srgb_processed,cv2.ROTATE_90_COUNTERCLOCKWISE)
    srgb_processed = cv2.flip(srgb_processed,0)


    # gt_image = ()
    patch_p = noisy_srgb[c:c + 796, c:c+ 796, :]
    psnr = PSNR(srgb_processed, patch_p)
    print(psnr)
    print(patch_p.shape)
    print(srgb_processed.shape)
    # cv2.imshow("roei_hagever_2",patch_p)
    cv2.imwrite("../../../../data/presnt_im/39noisy.png",patch_p)
    # cv2.imshow("roei_hagever_4",srgb_processed)
    cv2.imwrite("../../../../data/presnt_im/39noisy_pipeline.png",srgb_processed)
    # cv2.waitKey(0)






def load_cam_iso_nlf():

    cin = pd.read_csv('cam_iso_nlf.txt')
    cin = cin.drop_duplicates()
    cin = cin.set_index('cam_iso', drop=False)
    return cin


if __name__ == '__main__':
    main()

    # noisy_raw_path = os.path.abspath("../../../../data/raw/SIDD_Medium_Raw/Data/0039_002_IP_00100_00180_5500_L/0039_NOISY_RAW_010.MAT")
    # noisy_srgb_path = os.path.abspath("../../../../data/srgb/SIDD_Medium_Srgb/Data/0039_002_IP_00100_00180_5500_L/0039_NOISY_SRGB_010.PNG")
    # gt_path = os.path.abspath("../../../../data/srgb/SIDD_Medium_Srgb/Data/0039_002_IP_00100_00180_5500_L/0039_GT_SRGB_010.PNG")
    #
    # meta_data = os.path.abspath("../../../../data/raw/SIDD_Medium_Raw/Data/0039_002_IP_00100_00180_5500_L/0039_METADATA_RAW_010.MAT")
    # compare_pipelines(noisy_raw_path,noisy_srgb_path,meta_data,gt_path)

    # create_synt("../full_noise_set_15622.csv")
    # dict = get_best_lambdas("../lambda_and_shit.csv")
    # print(dict)
    # write_csv2(dict,"../best_lambdas_15622.csv")

    # main()

