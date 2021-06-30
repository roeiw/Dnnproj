import os
from PIL import Image
import io
import torch
import torchvision
import csv

small_patch_size = 80

print(os.getcwd())

path_to_data = "../../data/medium/SIDD_Medium_Srgb/Data/SIDD_Medium_Srgb/Data/"
path_to_patches = "../data/patches/medium/"
csv_path = path_to_patches + "image_csv.csv"

with open(csv_path, "w") as csv_in:
    writer=csv.writer(csv_in,lineterminator="\n")

    for f in os.listdir(path_to_data):
        if f == "patches":
            continue
        im_dir_path = path_to_patches + f
        if not os.path.exists(im_dir_path):
            os.mkdir(im_dir_path)
        for im in os.listdir(path_to_data + f):
            split = im.split("_")
            im_type = split[1]
            name = split[3]
            name = name.split(".")[0]
            img = Image.open(path_to_data+f+"/"+im)
            width = img.size[0]
            height = img.size[1]

            for i in range(40):
                for j in range(30):
                    w_offset = ((width/40)-small_patch_size)/2
                    h_offset = ((height/30)-small_patch_size)/2
                    small_patch = img.crop(((small_patch_size * i)+w_offset, (small_patch_size * j) + h_offset, small_patch_size * (i + 1)+w_offset,
                                            small_patch_size * (j + 1)+h_offset))
                    patch_path = im_dir_path+"/"+im_type+"_"+str(i)+"_"+str(j)+"_"+name
                    small_patch.save(patch_path+".PNG")
                    if im_type == "GT":
                        row=[]
                        row.append(patch_path)
                        writer.writerow(row)











