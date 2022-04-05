import h5py
import glob
import os
import cv2
import numpy as np
from scipy.io import loadmat
import rawpy
from pipeline_code.pipeline import run_pipeline
from pipeline_code.pipeline_util import get_visible_raw_image
from process_raw import DngFile



path = "../mats/raw_mats/images/0001_001_S6_00100_00060_3200_L/0001_GT_RAW_010.MAT"
path_meta = "../../mats/raw_mats/images/0001_001_S6_00100_00060_3200_L/0001_METADATA_RAW_010.MAT"

with h5py.File(path, 'r') as f:
                im = np.array(f.get('x')[:])
                # cv2.imshow("dng", im)
                # cv2.waitKey(0)
                im = (im*255).astype(np.uint8)
                print(im.max())
                print(im.min())

                DngFile.save("../mats/raw_mats/dngs/im1.dng",im,bit=12,pattern="GRBG")

