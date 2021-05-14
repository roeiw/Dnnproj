import os
from PIL import Image
import io
import torch
import torchvision
import csv

big_patch_size = 512
small_patch_size = 80

print(os.getcwd())

path_to_data = "../data/"
path_to_patches = "../data/patches/"
csv_path = path_to_data + "image_csv.csv"

with open(csv_path, "w") as csv_in:
    writer=csv.writer(csv_in,lineterminator="\n")

    for f in os.listdir(path_to_data):
        if f == "patches":
            continue
        im_dir_path = path_to_patches + f
        if not os.path.exists(im_dir_path):
            os.mkdir(im_dir_path)
        for im in os.listdir(path_to_data + f):
            im_type = im.split("_")[0]

            img = Image.open(path_to_data+f+"/"+im)
            for i in range(int(img.size[0]/big_patch_size)):
                for j in range(int(img.size[1]/big_patch_size)):
                    big_patch = img.crop((big_patch_size*i,big_patch_size*j,big_patch_size*(i+1),big_patch_size*(j+1)))
                    for m in range(int(big_patch.size[0] / small_patch_size)):
                        for l in range(int(big_patch.size[1] / small_patch_size) ):
                            small_patch = big_patch.crop((small_patch_size * i, small_patch_size * j, small_patch_size * (i + 1),
                                                          small_patch_size * (j + 1)))
                            patch_path = im_dir_path+"/"+im_type+"_"+str(i)+"_"+str(j)+"_"+str(m)+"_"+str(l)
                            small_patch.save(patch_path+".PNG")
                            if im_type == "GT":
                                row=[]
                                row.append(patch_path)
                                writer.writerow(row)










