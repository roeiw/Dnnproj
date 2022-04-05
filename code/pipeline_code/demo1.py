import glob
import os
import cv2
import numpy as np
import h5py
from scipy.io import loadmat
import rawpy
from pipeline_code.pipeline import run_pipeline
from pipeline_code.pipeline_util import get_visible_raw_image



path = "../../mats/raw_mats/images/0001_001_S6_00100_00060_3200_L/0001_GT_RAW_010.MAT"
path_meta = "../../mats/raw_mats/images/0001_001_S6_00100_00060_3200_L/0001_METADATA_RAW_010.MAT"

# with h5py.File(path, 'r') as f:
#                 im = np.array(f.get('x')[:])
# for path,subdirs,mats in os.walk("../../mats/raw_mats/images/"):
#
#     for mat in tqdm(mats):
#         if "GT" in mat:
#             with h5py.File(mat, 'r') as f:
#                 im = np.array(f.get('x')[:])
#                 rawpy.
#
#
#
#
bayer_id = 33422
bayer_tag_idx = 1
meta_mat=loadmat(path_meta)
meta_mat = meta_mat['metadata'][0, 0]
unknown_tags = meta_mat['UnknownTags']
if unknown_tags[bayer_tag_idx]['ID'][0][0][0] == bayer_id:
    bayer_pattern = unknown_tags[bayer_tag_idx]['Value'][0][0]
    print(bayer_pattern)
# print(meta_mat['UnknownTags'])
# # #np.array(f.get('x')[:])
# # with h5py.File(path_meta, 'r') as f:
# #     print(f)

#
# params = {
#     'output_stage': 'tone',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
#     'save_as': '.png',  # options: 'jpg', 'png', 'tif', etc.
#     'demosaic_type': 'menon2007',  # options: '', 'EA', 'VNG' 'menon2007',
#     'save_dtype': np.uint8
# }
#
# # processing a directory
# images_dir = '../../mats/raw_mats/dngs/'
# image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
# image = image_paths[0]
# image = get_visible_raw_image(image)
#
# for image_path in image_paths:
#     output_image = run_pipeline(image_path, params)
#     output_image_path = image_path.replace('.dng', '_{}'.format(params['output_stage']) +params['demosaic_type'] + params['save_as'])
#     max_val = 2 ** 16 if params['save_dtype'] == np.uint16 else 255
#     output_image = (output_image[..., ::-1] * 255).astype(params['save_dtype'])
#     if params['save_as'] == 'jpg':
#         cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
#     else:
#         cv2.imwrite(output_image_path, output_image)