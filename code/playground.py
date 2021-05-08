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

class Demo(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        #self.name = 'Demo'
        #self.scale = args.scale
        #self.idx_scale = 0
        self.train = False
        self.benchmark = False

        self.filelist = []
        # print(args.dir_demo)
        for f in os.listdir(args.dir_demo):
            # print("got in for")
            if f.find('.PNG') >= 0:
                print("got in if")
                self.filelist.append(os.path.join(args.dir_demo, f))
       # / print(len(self.filelist))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        print(filename)
        filename, _ = os.path.splitext(filename)
        lr = io.imread(self.filelist[idx])
        #lr = np.asfarray(lr)
        #lr = common.set_channel([lr], self.args.n_colors)[0]

        return lr

    def __len__(self):
        return len(self.filelist)

    # def set_scale(self, idx_scale):
    #     self.idx_scale = idx_scale
    #


datadata = Demo(args)
image = datadata[0]
# print(image[0].numpy())
# torch.
plt.imshow(image)
plt.show()