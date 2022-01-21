import torchvision
import os
import math
import time
import datetime
from functools import reduce

import matplotlib
from torchvision import models
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kornia
import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from vgg16 import Vgg16
import lpips
from pytorch_msssim import ms_ssim



def msssim(pred_batch,gt_batch):
    ms_ssim_loss = 1-ms_ssim(pred_batch,gt_batch,data_range=1,win_size=7)
    return ms_ssim_loss


def rgb2Ycrcb(input):
    input = input.cuda().reshape(3,-1)

    Y_vector = Variable(torch.FloatTensor([0.299, 0.587, 0.114]),requires_grad=False).cuda().reshape(1,3)
    output = torch.matmul(Y_vector, input).cuda()
    return output

def Yloss(pred_batch,gt_batch):
    batch = pred_batch.shape
    # batch_loss = torch.empty(size = batch)
    lab_batch_pred = torch.empty(size=(batch[0],batch[2]*batch[3]))
    lab_batch_gt = torch.empty(size=(batch[0],batch[2]*batch[3]))
    for i in range(batch[0]):
        lab_batch_pred[i] = rgb2Ycrcb(pred_batch[i, :, :, :])
        lab_batch_gt[i] = rgb2Ycrcb(gt_batch[i, :, :, :])



    # lab_batch_pred = kornia.color.rgb_to_lab(pred_batch)
    # lab_batch_gt = kornia.color.rgb_to_lab(gt_batch)

    loss = torch.abs(lab_batch_pred - lab_batch_gt)
    return loss.mean()


class Y_L1(torch.nn.Module):
    def __init__(self):
        super(Y_L1,self).__init__()
        self.loss = torch.nn.L1Loss()
        self.alpha = 0.5
    def forward(self,pred,gt):
        Y_loss = Yloss(pred,gt)
        L1 = self.loss(pred,gt)
        return (self.alpha*Y_loss)+((1-self.alpha)*L1)


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale*(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def get_features(image, model, layers):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """

    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image

    image = image.cuda()
    features = []

    for layer_num, layer in enumerate(model):
        # activation of the layer will stored in x
        image = layer(image)
        # appending the activation of the selected layers and return the feature array
        if (str(layer_num) in layers):
            features.append(image)

    # features = torch.
    # x =
    # model._modules is a dictionary holding each module in the model
    # print(features.shape)
    # print(features.shape)
    return features


def create_vgg19():
    vgg = models.vgg19(pretrained=True).features[:29]


    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)
    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg.to(device)
    # print(vgg)
    return vgg

class contentLoss(torch.nn.Module):
    def __init__(self, layers=None):

        super(contentLoss, self).__init__()
        net = create_vgg19()
        self.model = net
        self.req_features = ['0', '5', '10', '19', '28']

    def forward(self, pred_batch, gt_batch):
        batch_size = pred_batch.shape[0]
        # num_of_layers = len(self.layers)
        total_loss = 0

        gt_features = get_features(gt_batch[:,:,:,:], self.model, self.req_features)
        pred_features = get_features(pred_batch[:,:,:,:], self.model, self.req_features)
        # print(len(gt_features))
        feature_len = len(gt_features)
        for i in range(feature_len):
            total_loss += torch.mean((gt_features[i]-pred_features[i])**2)

        return total_loss/feature_len

def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)*2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)


    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out


def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

class content_lab_loss(torch.nn.Module):
    def __init__(self):
        super(content_lab_loss,self).__init__()
        self.model=Vgg16(requires_grad=False)
        self.loss = torch.nn.MSELoss()
        self.alpha = 1
    def forward(self,pred,gt):
        ContentLoss_Val = content_loss(self.model,self.loss,pred,gt)
        LabLoss_Val = LabLoss_L2(pred,gt)
        return (self.alpha*ContentLoss_Val)+((1-self.alpha)*LabLoss_Val)


def content_loss(model, loss_function, pred_batch, gt_batch):
    gt_features = model(gt_batch).relu2_2
    pred_features = model(pred_batch).relu2_2
    # print(str(gt_features.shape) + "type of gt featues and val")
    return loss_function(pred_features,gt_features)

def lpips_func():

    return lpips.LPIPS(net='vgg')

def msssim_L1(pred_batch, gt_batch):
    msssim_loss = msssim(pred_batch,gt_batch)
    L1_loss = torch.abs(pred_batch-gt_batch).mean()
    return 0.5*(msssim_loss+L1_loss)

def LabLoss_L1 (pred_batch, gt_batch):
    batch = pred_batch.shape
    # batch_loss = torch.empty(size = batch)
    lab_batch_pred = torch.empty(size = batch)
    lab_batch_gt = torch.empty(size = batch)
    for i in range(batch[0]):
        lab_batch_pred[i] = rgb_to_lab(pred_batch[i,:,:,:])
        lab_batch_gt[i] = rgb_to_lab(gt_batch[i,:,:,:])

    # lab_batch_pred = kornia.color.rgb_to_lab(pred_batch)
    # lab_batch_gt = kornia.color.rgb_to_lab(gt_batch)

    loss = torch.abs(lab_batch_pred-lab_batch_gt)
    return loss.mean()

def LabLoss_L2 (pred_batch, gt_batch):
    # lab_batch_pred = kornia.color.rgb_to_lab(pred_batch)
    # lab_batch_gt = kornia.color.rgb_to_lab(gt_batch)
    batch = pred_batch.shape
    # batch_loss = torch.empty(size = batch)
    lab_batch_pred = torch.empty(size = batch)
    lab_batch_gt = torch.empty(size = batch)
    for i in range(batch[0]):
        lab_batch_pred[i] = rgb_to_lab(pred_batch[i,:,:,:])
        lab_batch_gt[i] = rgb_to_lab(gt_batch[i,:,:,:])


    return torch.mean((lab_batch_gt - lab_batch_pred) ** 2)



def rgb_to_lab(img):
    """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    :param img: image to be adjusted
    :returns: adjusted image
    :rtype: Tensor
    """

    img = img.permute(2, 1, 0)
    shape = img.shape
    img = img.contiguous()
    img = img.view(-1, 3)

    # all color transformation references are from brucelindbloom.com

    # inverse srgb companding. 2.2 exponent also possible
    img = ((img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img, min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()).cuda()

    # RGB -> XYZ (not xyz!)
    # checked with lindbloom sRGB D65 matrix, slightly different. matrix from lindbloom:
    # [0.4124564  0.3575761  0.1804375]
    # [0.2126729  0.7151522  0.0721750]
    # [0.0193339  0.1191920  0.9503041]
    rgb_to_xyz = Variable(torch.FloatTensor([[0.412453, 0.212671, 0.019334], [0.357580, 0.715160, 0.119193], [0.180423, 0.072169, 0.950227]]), requires_grad = False).cuda()

    # RGB -> XYZ step
    img = torch.matmul(img, rgb_to_xyz)

    # XYZ -> xyz step with reference white (0.950456, 1, 1.088754)
    # closest to D65 chromatic adaption matrix
    img = torch.mul(img, Variable(torch.FloatTensor([1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad = False).cuda())

    # third root of CIE epsilon of 0.008856
    epsilon = 6 / 29

    # xyz -> Lab
    img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + (torch.clamp(img, min=0.0001) ** (1.0 / 3.0) * img.gt(epsilon ** 3).float())
    fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],[116.0, -500.0, 200.0], [0.0, 0.0, -200.0]]), requires_grad = False).cuda()
    img = torch.matmul(img, fxfyfz_to_lab) + Variable(torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad = False).cuda()

    img = img.view(shape)
    img = img.permute(2, 1, 0)

    '''
    L_chan: black and white with input range [0, 100]
    a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
    [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
    '''
    img[0, :, :] = img[0, :, :] / 100
    img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
    img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

    # ??? - check this
    img[(img != img).detach()] = 0
    img = img.contiguous()

    return img.cuda()

def lab_to_rgb(img):
        """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor
        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)
        img_copy = img.clone()

        img_copy[:, 0] = img[:, 0] * 100
        img_copy[:, 1] = ((img[:, 1] * 2) - 1) * 110
        img_copy[:, 2] = ((img[:, 2] * 2) - 1) * 110

        img = img_copy.clone().cuda()
        del img_copy

        lab_to_fxfyfz = Variable(torch.FloatTensor([[1 / 116.0, 1 / 116.0, 1 / 116.0], [1 / 500.0, 0, 0], [0, 0, -1 / 200.0]]), requires_grad = False).cuda()

        img = torch.matmul(img + Variable(torch.cuda.FloatTensor([16.0, 0.0, 0.0])), lab_to_fxfyfz)

        epsilon = 6.0 / 29.0

        img = (((3.0 * epsilon ** 2 * (img - 4.0 / 29.0)) * img.le(epsilon).float()) + ((torch.clamp(img, min=0.0001) ** 3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, Variable(torch.cuda.FloatTensor([0.950456, 1.0, 1.088754])))

        xyz_to_rgb = Variable(torch.FloatTensor([[3.2404542, -0.9692660, 0.0556434], [-1.5371385, 1.8760108, -0.2040259], [-0.4985314, 0.0415560, 1.0572252]]), requires_grad = False).cuda()
        img = torch.matmul(img, xyz_to_rgb)
        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img, min=0.0001) ** (1 / 2.4) * 1.055) - 0.055) * img.gt(0.0031308).float()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        img = img.contiguous()
        img[(img != img).detach()] = 0

        return img

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.module.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'Denoise on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, noise in enumerate(self.args.noise_g):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Noise {}'.format(noise)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

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

def calc_psnr(sr, hr, rgb_range):
    diff = (sr - hr).data.div(rgb_range)

    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''

    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)

def get_loss(args):
    loss_type = args.loss
    if loss_type == 'MSE':
        loss_function = nn.MSELoss()
    elif loss_type == 'L1':
        loss_function = nn.L1Loss()
    elif loss_type.find('Lab_L1') >= 0:
        loss_function = LabLoss_L1

    elif loss_type.find('Lab_L2') >= 0:
        loss_function = LabLoss_L2

    return loss_function
# def make_optimizer(args, my_model):
#     trainable = filter(lambda x: x.requires_grad, my_model.parameters())
#
#     if args.optimizer == 'SGD':
#         optimizer_function = optim.SGD
#         kwargs = {'momentum': args.momentum}
#     elif args.optimizer == 'ADAM':
#         optimizer_function = optim.Adam
#         kwargs = {
#             'betas': (args.beta1, args.beta2),
#             'eps': args.epsilon
#         }
#     elif args.optimizer == 'RMSprop':
#         optimizer_function = optim.RMSprop
#         kwargs = {'eps': args.epsilon}
#
#     kwargs['lr'] = args.lr
#     kwargs['weight_decay'] = args.weight_decay
#
#     return optimizer_function(trainable, **kwargs)

# def make_scheduler(args, my_optimizer):
#     if args.decay_type == 'step':
#         scheduler = lrs.StepLR(
#             my_optimizer,
#             step_size=args.lr_decay,
#             gamma=args.gamma
#         )
#     elif args.decay_type.find('step') >= 0:
#         milestones = args.decay_type.split('_')
#         milestones.pop(0)
#         milestones = list(map(lambda x: int(x), milestones))
#         scheduler = lrs.MultiStepLR(
#             my_optimizer,
#             milestones=milestones,
#             gamma=args.gamma
#         )
#
#     return scheduler

def imshow(data):
    gt = data['GT_image']
    nois = data['NOISY_image']
    name = data['image_name']
    gt = gt.permute((0,3,1,2))
    inp = torchvision.utils.make_grid(gt)
    nois = nois.permute((0,3,1,2))
    inp2 = torchvision.utils.make_grid(nois)
    inp = inp.numpy().transpose((1, 2, 0))
    inp2 = inp2.numpy().transpose((1, 2, 0))
    fig1 = plt.figure(1)
    plt.imshow(inp)
    fig1.show()
    fig2 = plt.figure(2)
    plt.imshow(inp2)
    fig2.show()