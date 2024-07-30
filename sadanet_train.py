import sys
import glob
import numpy.matlib
import numpy as np
import cv2
import re
from termcolor import colored
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

#import torchvision  #for deformconv2

from osgeo import gdal

import random
import configparser
from typing import Tuple
import matplotlib.pyplot as plt

from guided_filter_pytorch.guided_filter import GuidedFilter


#used as prefix for saved weights
model_name = 'grss'

#folder with training data
input_folder = '/media/HDD/TrainingsData/MB_H/trainingHDisp/*/'

save_folder_branch = './weights/branch/'
save_folder_simb = './weights/simB/'

out_folder = './Out/'

#needs to be odd
#size of patch-crops fed into the network
patch_size = 11#11
ps_h = int(patch_size/2)

#range for offset of o_neg
r_low = 1
r_high = 25


Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
cos = torch.nn.CosineSimilarity()

num_feat_branch = 100#60  #46
k = 3
p = 1

class SiameseBranch64(nn.Module):
    def __init__(self,img_ch=1):
        super(SiameseBranch64,self).__init__()

        self.Tanh = nn.Tanh() 
        self.Conv1 = nn.Conv2d(img_ch, num_feat_branch, kernel_size = k,stride=1,padding = p,dilation = 1, bias=True)      
        self.Conv2 = nn.Conv2d(num_feat_branch, num_feat_branch, kernel_size = k,stride=1,padding = p,dilation = 1, bias=True)
        self.Conv3 = nn.Conv2d(num_feat_branch, num_feat_branch, kernel_size = k,stride=1,padding = p,dilation = 1, bias=True)
        self.Conv4 = nn.Conv2d(num_feat_branch, 60, kernel_size = k,stride=1,padding = p,dilation = 1, bias=True)

    def forward(self,x_in):

        x1 = self.Conv1(x_in) 
        x1 = self.Tanh(x1)
        
        x2 = self.Conv2(x1) 
        x2 = self.Tanh(x2)

        x3 = self.Conv3(x2) 
        x3 = self.Tanh(x3)
        
        x4 = self.Conv4(x3) 

        return x4

    
branch = SiameseBranch64()
branch = branch.cuda()




num_feat_simb = 80#50  #45 
class SimMeasTanh(nn.Module): #needs to be 2*60
    def __init__(self,img_ch=2*60):
        super(SimMeasTanh,self).__init__()

        self.tanh = nn.Tanh() 

        self.Conv1 = nn.Conv2d(img_ch, num_feat_simb, kernel_size = 3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv2 = nn.Conv2d(num_feat_simb, num_feat_simb, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv3 = nn.Conv2d(num_feat_simb, num_feat_simb, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv4 = nn.Conv2d(num_feat_simb, num_feat_simb, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv5 = nn.Conv2d(num_feat_simb, 1, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)

    def forward(self,x_in):
 
        x1 = self.Conv1(x_in) 
        x1 = self.tanh(x1)
        

        x2 = self.Conv2(x1) 
        x2 = self.tanh(x2)
        
        x3 = self.Conv3(x2) 
        x3 = self.tanh(x3)
        
        x4 = self.Conv4(x3) 
        x4 = self.tanh(x4) 
                        
        x5 = self.Conv5(x4)

        return x5


simB = SimMeasTanh()
simB = simB.cuda()


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def loadKitti2015(input_folder):

    left_filelist = glob.glob(input_folder + 'image_2/*.png')
    right_filelist = glob.glob(input_folder + 'image_3/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc_0/*.png')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)

    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)


    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)

    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)

    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)
   
    inters_list = list(inters_list)
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(inters_list)):
        
        left_im = input_folder + 'image_2/' + inters_list[i]
        right_im = input_folder + 'image_3/' + inters_list[i]
        disp_im =  input_folder + 'disp_noc_0/' + inters_list[i] 
       
        cur_left = cv2.imread(left_im)
        cur_right = cv2.imread(right_im)
        cur_disp = cv2.imread(disp_im)
        
        cur_disp = np.mean(cur_disp,axis=2) 
        #set 0 (invalid) to inf to be same as MB for Batchloader
        cur_disp[np.where(cur_disp == 0.0)] = np.inf
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list

def loadMB(input_folder):
    
    left_filelist = glob.glob(input_folder + '/im0.png')
    right_filelist = glob.glob(input_folder + '/im1.png')
    disp_filelist = glob.glob(input_folder + '/disp0GT.pfm')
    calib_filelist = glob.glob(input_folder + '/calib.txt')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)
    calib_filelist = sorted(calib_filelist)
    
    left_list = []
    right_list = []
    disp_list = []
    maxdisp_list = []
    s_name_list = []
    
    for i in range(0,len(left_filelist)):
        
        cur_left = cv2.imread(left_filelist[i])
        cur_right = cv2.imread(right_filelist[i])        
        
        cur_disp,_ = readPFM(disp_filelist[i])
        
        cur_disp[np.isnan(cur_disp)] = 0
        cur_disp[np.isinf(cur_disp)] = 0
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
        f = open(calib_filelist[i],'r')
        calib = f.read()
        max_disp = int(calib.split('\n')[6].split("=")[1])
        
        s_name = left_filelist[i].split('/')[-2]
        
        maxdisp_list.append(max_disp)
        s_name_list.append(s_name)
        
        
    return left_list, right_list, disp_list, maxdisp_list, s_name_list

def _compute_binary_kernel(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = _compute_binary_kernel(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median

# functiona api
def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    See :class:`~kornia.filters.MedianBlur` for details.
    """
    return MedianBlur(kernel_size)(input)


def filterCostVolMedianPyt(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol = cost_vol.unsqueeze(0)
    
    for disp in range(d):

        cost_vol[:,disp,:,:] = median_blur(cost_vol[:,disp,:,:].unsqueeze(0), (5,5))
        
    return torch.squeeze(cost_vol)

def filterCostVolBilatpyt(cost_vol,left):
    
    leftT = Variable(Tensor(left))
    leftT = leftT.unsqueeze(0).unsqueeze(0)

    d,h,w = cost_vol.shape  
    
    f = GuidedFilter(8,10).cuda()  #10 #0.001
    
    for disp in range(d):
        cur_slice =  cost_vol[disp,:,:]
        cur_slice = cur_slice.unsqueeze(0).unsqueeze(0)
        
        inputs = [leftT, cur_slice]

        test = f(*inputs)
        cost_vol[disp,:,:] = np.squeeze(test)
        
    return cost_vol

def createCostVol(branch, simB,left_im,right_im,max_disp, filtered):
        
    a_h, a_w,c = left_im.shape

    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)
    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])

    with torch.no_grad():

        left_imT = Variable(Tensor(left_im.astype(np.uint8)))
        right_imT = Variable(Tensor(right_im.astype(np.uint8)))
        
        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)        
        

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        cost_volT = Variable(Tensor(cost_vol))

        #0 => max_disp => one less disp!
        #python3 apparently cannot have 0 here for disp: right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
        for disp in range(0,max_disp+1):

            if(disp == 0):
                
                sim_score = simB(torch.cat((left_feat, right_feat),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score)                
            else:
                right_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)                      
                right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
                right_appended = torch.cat([right_shift,right_feat],3)

                _,f,h_ap,w_ap = right_appended.shape
                right_shifted[:,:,:,:] = right_appended[:,:,:,:(w_ap-disp)]
                sim_score = simB(torch.cat((left_feat, right_shifted),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score)              

    return cost_volT

def createCostVolRL(branch, simB, left_im,right_im,max_disp, filtered):

    a_h, a_w,c = left_im.shape

    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)
    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])

    with torch.no_grad():
        
        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)        
        
        left_feat = branch(left_imT)
        right_feat = branch(right_imT)


        _,f,h,w = left_feat.shape
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        
        cost_volT = Variable(Tensor(cost_vol))

        for disp in range(0,max_disp+1):

            if(disp == 0):
                sim_score = simB(torch.cat((left_feat, right_feat),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score) 
            else:    
                left_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)
                left_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)
                left_appended = torch.cat([left_feat,left_shift],3)

                _,f,h_ap,w_ap = left_appended.shape
                left_shifted[:,:,:,:] = left_appended[:,:,:,disp:w_ap]
            
                sim_score = simB(torch.cat((left_shifted, right_feat),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score)
                
    return cost_volT

def createCostVolAllTogetherSimB(left_im,right_im,max_disp):    
    
    a_h, a_w,c = left_im.shape
    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)
    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])
    
    with torch.no_grad():

        left_imT = Variable(Tensor(left_im.astype(np.uint8)))
        right_imT = Variable(Tensor(right_im.astype(np.uint8)))

        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_volL = np.zeros((max_disp+1,a_h,a_w))
        cost_volLT = Variable(Tensor(cost_volL))   
    
        cost_volR = np.zeros((max_disp+1,a_h,a_w))
        cost_volRT = Variable(Tensor(cost_volR))   
    
        #0 => max_disp => one less disp!
        for disp in range(0,max_disp+1):
            
            if(disp == 0):
                sim_score_l = simB(torch.cat((left_feat, right_feat),dim=1))
                sim_score_r = simB(torch.cat((left_feat, right_feat),dim=1))

                cost_volRT[disp,:,:] = torch.squeeze(sim_score_l) 
                cost_volLT[disp,:,:] = torch.squeeze(sim_score_r) 

            else:
                
                left_shifted = torch.roll(left_feat, -disp, dims = 3) 
                left_shifted[:,:,:,w-disp:w] = 0
                sim_score_right = simB(torch.cat((left_shifted, right_feat),dim=1))
                cost_volRT[disp,:,:] = torch.squeeze(sim_score_right) 
                
    
                right_shifted = torch.roll(right_feat, disp, dims = 3) 
                right_shifted[:,:,:,0:disp] = 0
                sim_score_left = simB(torch.cat((left_feat, right_shifted),dim=1))
                cost_volLT[disp,:,:] = torch.squeeze(sim_score_left) 
                

    return cost_volLT, cost_volRT

def TestImage(branch, simB, fn_left, fn_right, max_disp, filtered, lr_check, dataset):
    
    left = cv2.imread(fn_left)
    right = cv2.imread(fn_right)
    
    disp_map = []
    
    if(filtered):
        
        cost_volLT, cost_volRT = createCostVolAllTogetherSimB(left,right,max_disp, True)
        
        cost_vol_filteredn = filterCostVolBilatpyt(cost_volLT,left)
        cost_vol_filteredn = np.squeeze(cost_vol_filteredn.cpu().data.numpy())                
        disp = np.argmax(cost_vol_filteredn, axis=0) 
        
        #del cost_vol
        #del cost_vol_filteredn
        #torch.cuda.empty_cache()              
        
        if(lr_check):
            #cost_vol_RL = createCostVolRL(branch, simB,left,right,max_disp, True)
            
            cost_vol_RL_fn = filterCostVolBilatpyt(cost_volRT,right)
            cost_vol_RL_fn = np.squeeze(cost_vol_RL_fn.cpu().data.numpy())
            
            disp_map_RL = np.argmax(cost_vol_RL_fn, axis=0)  
            disp_map = LR_Check(disp.astype(np.float32), disp_map_RL.astype(np.float32), dataset)
            
            #del cost_vol_RL
            #del cost_vol_RL_fn
            #torch.cuda.empty_cache()              
        
    else:
        
        cost_volLT, cost_volRT = createCostVolAllTogetherSimB(left,right,max_disp, True)
        cost_vol = np.squeeze(cost_volLT.cpu().data.numpy())
        disp = np.argmax(cost_vol, axis=0)        
        
        if(lr_check):
            
            #cost_vol_RL = createCostVolRL(branch, simB,left,right,max_disp, False)
            cost_vol_RL = np.squeeze(cost_volRT.cpu().data.numpy())
            disp_map_RL = np.argmax(cost_vol_RL, axis=0)       
            disp_map = LR_Check(disp.astype(np.float32), disp_map_RL.astype(np.float32), 'MB')
    if(lr_check):
        return disp_map, disp, disp_map_RL
    else:
        return disp
    

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    scale = -scale

    file.write('%f\n'.encode() % scale)
    image.tofile(file)
    
def FillIncons(mask, disp):

    #limit for consistent point search
    max_search = 30
    w = mask.shape[1]
    h = mask.shape[0] 
    
    #BG
    idc = np.argwhere(np.isnan(disp))    
    for curnan in range(len(idc)):
        curnanh = idc[curnan][0]
        curnanw = idc[curnan][1]        
        if(mask[curnanh,curnanw] == 0):
            
            #whole scanline is nan => disp is 0
            if(all(np.isnan(disp[curnanh,:]))):
                #hole line set to 0!
                disp[curnanh,:] = 0.0
            #all px to the left are NaN
            if(all(np.isnan(disp[curnanh,0:curnanw]))):
                #go to the right
                curw = curnanw
                fill = 0
                while(np.isnan(disp[curnanh,curw]) and mask[curnanh,curnanw] == 0):
                    curw = curw +1
                    fill = disp[curnanh,curw]
                disp[curnanh,curnanw] = fill  
            #else go left
            else:
                curw = curnanw
                fill = 0
                while(np.isnan(disp[curnanh,curw]) and mask[curnanh,curnanw] == 0):
                    curw = curw -1
                    fill = disp[curnanh,curw]
                disp[curnanh,curnanw] = fill 
    #FG
    idcFG = np.argwhere(np.isnan(disp))
    for curnan in range(len(idcFG)):
        
        curnanh = idcFG[curnan][0]
        curnanw = idcFG[curnan][1]
      
        left = 0
        right = 0
        above = 0
        under = 0

        r_above = 0
        l_above = 0
        r_under = 0
        l_under = 0      
        
        if(curnanw == 0):
            left = 0
        else:
            left = int(disp[curnanh,curnanw-1])
        counter = 0                                    
        while(np.isnan(disp[curnanh,curnanw+counter])):
            counter = counter +1                       
            if((curnanw+counter) >= w or counter >= max_search):
                right = 0
                break
            right = disp[curnanh,curnanw+counter]
        counter = 0                                    
        while(np.isnan(disp[curnanh+counter,curnanw])):
            counter = counter +1                       
            if((curnanh+counter) >= h or counter >= max_search):
                above = 0
                break       
            above = disp[curnanh+counter,curnanw]
        if(curnanh == 0):
            under = 0
        else:
            under = disp[curnanh-1,curnanw]
        
        counter = 0                                    
        while(np.isnan(disp[curnanh+counter,curnanw+counter])):
            counter = counter +1
            if((curnanh+counter) >= h or counter >= max_search):
                r_above = 0
                break
            if((curnanw+counter) >= w):
                r_above = 0
                break                        
            r_above = disp[curnanh+counter,curnanw+counter]   
        
        if(curnanh == 0 or curnanw == 0):
            l_under = 0
        else:
            l_under = disp[curnanh-1,curnanw-1]  
        
        counter = 0      
        while(np.isnan(disp[curnanh+counter,curnanw-counter])):
            counter = counter +1
            if((curnanh+counter) >= h):
                l_above = 0
                break
            if((curnanw-counter) <= 0 or counter >= max_search):
                l_above = 0
                break
            l_above = disp[curnanh+counter,curnanw-counter]

        if(curnanh == 0 or curnanw >= w-1):
            r_under = 0
        else:
            r_under = disp[curnanh-1,curnanw+1]
        
        fill = np.median([left,right,above,under,r_above,l_above,r_under,l_under])
        disp[curnanh,curnanw] = fill
    return disp

def calcEPE(disp, gt_fn):
    
    gt = gt_fn

    gt[np.where(gt == np.inf)] = -100
    #for loadmb
    gt[np.where(gt == 0)] = -100
    
    mask = gt > 0

    disp = disp[mask]
    gt = gt[mask]        

    nr_px = len(gt)


    abs_error_im = np.abs(disp - gt)

    five_pe = (float(np.count_nonzero(abs_error_im >= 5.0) ) / nr_px) * 100.0  
    four_pe = (float(np.count_nonzero(abs_error_im >= 4.0) ) / nr_px) * 100.0  
    three_pe = (float(np.count_nonzero(abs_error_im >= 3.0) ) / nr_px) * 100.0  
    two_pe = (float(np.count_nonzero(abs_error_im >= 2.0) ) / nr_px) * 100.0        
    one_pe = (float(np.count_nonzero(abs_error_im >= 1.0) ) / nr_px) * 100.0        
    pf_pe = (float(np.count_nonzero(abs_error_im >= 0.5) ) / nr_px) * 100.0  
        
    return five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe


def getBatch(gt_cons_cpy):
    
    batch_xl = np.zeros((batch_size,1,patch_size,patch_size))
    batch_xr_pos = np.zeros((batch_size,1,patch_size,patch_size))
    batch_xr_neg = np.zeros((batch_size,1,patch_size,patch_size))
    
    for el in range(batch_size):
        
        
        if(el % 10 == 0):
            
            ridx = np.random.randint(0,len(gt_cons_cpy),1)
            
            left_im = left_list[ridx[0]]
            right_im = right_list[ridx[0]]
            
            #left_im = right_list_train[ridx[0]]
            #right_im = left_list_train[ridx[0]]
            
            
            gt_im = gt_cons_cpy[ridx[0]]
            #left = left_list[ridx[0]]
        
        
        h,w = left_im.shape
        r_h = 0
        r_w = 0
        d = 0
#        print('Draw for random position')
        #also check height! should not draw corner pixels!!
        while True:
            
            r_h = random.sample(range(ps_h,h-(ps_h+1)), 1)
            r_w = random.sample(range(ps_h,w-(ps_h+1)),1)   
            
            #wrong direction right?
            
            if(not np.isnan(gt_im[r_h,r_w])):
                #not int and not round!
                d = gt_im[r_h,r_w]
                if((r_w[0]+ps_h+d+1) <= w):
                     if((r_w[0]-(ps_h+1)+d-1) >= 0):
                        break
        
        d = int(np.round(gt_im[r_h,r_w]))
                
        cur_left = left_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), r_w[0]-ps_h:r_w[0]+(ps_h+1)]
        #choose offset
        
        cur_right_pos = right_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), (r_w[0]-ps_h+d):(r_w[0]+(ps_h+1)+d)]

        
        #should not be too close to real match!
        o_neg = 0
        while True:
            #range 6-8??? range(2,6)
            o_neg = random.sample(range(r_low,r_high), 1)
            if np.random.randint(-1, 1) == -1:
                o_neg = -o_neg[0]
            else:
                o_neg = o_neg[0]
            #try without d-+1   and(o_neg != (d-1)) and(o_neg != (d+1))
            if((o_neg != d) and ((r_w[0]-ps_h+d+o_neg) > 0)  and ((r_w[0]+(ps_h+1)+d+o_neg) < w)):
                break
        
        
        cur_right_neg = right_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), (r_w[0]-ps_h+d+o_neg):(r_w[0]+(ps_h+1)+d+o_neg)]        
        
        batch_xl[el,:,:,:] =  cur_left
        batch_xr_pos[el,:,:,:] = cur_right_pos
        batch_xr_neg[el,:,:,:] = cur_right_neg
            
    return batch_xl, batch_xr_pos, batch_xr_neg#, batch_disp_xl



def my_hinge_loss(s_p, s_n):
    margin = 0.2
    relu = torch.nn.ReLU()
    relu = relu.cuda()
    loss = relu(-((s_p - s_n) - margin))

    return loss

def createCostVolBranch(branch, simB,left_im,right_im,max_disp):
    
    
    k_s = 19
    p = 9
    
    w = np.ones((k_s,k_s)).astype(np.float32)
    
    weights = Tensor(w)
    weights = weights.view(1, 1, k_s, k_s).repeat(1, 1, 1, 1)
        
    a_h, a_w,c = left_im.shape

    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)
    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])

    with torch.no_grad():

        left_imT = Variable(Tensor(left_im.astype(np.uint8)))
        right_imT = Variable(Tensor(right_im.astype(np.uint8)))
        
        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)        
        

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        cost_volT = Variable(Tensor(cost_vol))

        #0 => max_disp => one less disp!
        #python3 apparently cannot have 0 here for disp: right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
        for disp in range(0,max_disp+1):

            if(disp == 0):
                
                sim_score = cos(left_feat, right_feat)
                cost_volT[disp,:,:] = torch.squeeze(F.conv2d(torch.squeeze(sim_score).unsqueeze(0).unsqueeze(0), weights,padding = p))                
            else:
                right_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)                      
                right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
                right_appended = torch.cat([right_shift,right_feat],3)

                _,f,h_ap,w_ap = right_appended.shape
                right_shifted[:,:,:,:] = right_appended[:,:,:,:(w_ap-disp)]
                sim_score = cos(left_feat, right_shifted)
                cost_volT[disp,:,:] = torch.squeeze(F.conv2d(torch.squeeze(sim_score).unsqueeze(0).unsqueeze(0), weights,padding = p))              

    return cost_volT

def createCostVolRLBranch(branch, simB, left_im,right_im,max_disp):

    
    k_s = 19
    p = 9
    
    w = np.ones((k_s,k_s)).astype(np.float32)
    
    weights = Tensor(w)
    weights = weights.view(1, 1, k_s, k_s).repeat(1, 1, 1, 1)
    
    a_h, a_w,c = left_im.shape

    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)

    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])

    with torch.no_grad():
        
        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)        
        
        left_feat = branch(left_imT)
        right_feat = branch(right_imT)


        _,f,h,w = left_feat.shape
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        
        cost_volT = Variable(Tensor(cost_vol))

        for disp in range(0,max_disp+1):

            if(disp == 0):
                sim_score = cos(left_feat, right_feat)
                cost_volT[disp,:,:] = torch.squeeze(F.conv2d(torch.squeeze(sim_score).unsqueeze(0).unsqueeze(0), weights,padding = p)) 
            
            else:    
                left_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)
                left_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)
                left_appended = torch.cat([left_feat,left_shift],3)

                _,f,h_ap,w_ap = left_appended.shape
                left_shifted[:,:,:,:] = left_appended[:,:,:,disp:w_ap]
            
                sim_score = cos(left_shifted, right_feat)
                cost_volT[disp,:,:] = torch.squeeze(F.conv2d(torch.squeeze(sim_score).unsqueeze(0).unsqueeze(0), weights,padding = p))
                
    return cost_volT


#even further improve this by using pytorch!
def LR_Check(first_output, second_output, dataset):    
    
    h,w = first_output.shape
        
    line = np.array(range(0, w))
    idx_arr = np.matlib.repmat(line,h,1)    
    
    dif = idx_arr - first_output
    
    first_output[np.where(dif <= 0)] = 0
    
    first_output = first_output.astype(int)
    second_output = second_output.astype(int)
    dif = dif.astype(int)
    
    second_arr_reordered = np.array(list(map(lambda x, y: y[x], dif, second_output)))
    
    dif_LR = np.abs(second_arr_reordered - first_output)
    first_output[np.where(dif_LR >= 1.1)] = 0
    
    if(dataset == 'MB'):
        first_output[np.where(first_output <= 15.0)] = 0
        
    
    first_output = first_output.astype(np.float32)
    first_output[np.where(first_output == 0.0)] = np.nan
    
        
    return first_output

def RL_Check(first_output, second_output):    
    
    h,w = first_output.shape
        
    line = np.array(range(0, w))
    idx_arr = np.matlib.repmat(line,h,1)    
    
    dif = idx_arr + first_output
    
    dif[np.where(dif >= w)] = 0
    
    first_output = first_output.astype(int)
    second_output = second_output.astype(int)
    dif = dif.astype(int)    
    
    second_arr_reordered = np.array(list(map(lambda x, y: y[x], dif, second_output)))
    
    dif_RL = np.abs(second_arr_reordered - first_output)
    first_output[np.where(dif_RL >= 1.1)] = 0

    first_output = first_output.astype(np.float32)
    first_output[np.where(first_output == 0.0)] = np.nan
        
    return first_output

def getGT(epoch, avg_err):
    nr = 0
    gt_newlist = []
    disp_newlist = []
    nr_incons_tot = 0.0
    
    
    avg_five_pe = 0.0
    avg_four_pe = 0.0
    avg_three_pe = 0.0
    avg_two_pe = 0.0
    avg_one_pe = 0.0
    avg_pf_pe = 0.0      
    
    
    #samples problem!!!! Fix that!!!
    for i in range(0, len(left_list)):

        max_disp = max_disp_list[i]
        s_name = s_name_list[i]
        
        gt = gt_list[i]
        
        cost_vol = createCostVolBranch(branch, simB,left_list[i],right_list[i],max_disp)
        cost_volRL = createCostVolRLBranch(branch, simB,left_list[i],right_list[i],max_disp)

        disp = np.argmax(cost_vol.cpu().data.numpy(), axis=0) 
        dispRL = np.argmax(cost_volRL.cpu().data.numpy(), axis=0) 
        
        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp, gt)

        avg_five_pe = avg_five_pe + five_pe
        avg_four_pe = avg_four_pe +  four_pe
        avg_three_pe = avg_three_pe + three_pe
        avg_two_pe = avg_two_pe + two_pe
        avg_one_pe = avg_one_pe + one_pe
        avg_pf_pe = avg_pf_pe + pf_pe      

        #writePFM(out_folder +  s_name + '%06d_e%06f.pfm' %(epoch,two_pe),disp.astype(np.float32))
                
        #LR_Check sets certain values of disp to 0, should probably be a copy!
        disp_s = LR_Check(disp, dispRL, 'MB')

        #writePFM(out_folder +  s_name + '%06d_e%06f.pfm' %(epoch,two_pe),disp.astype(np.float32))
        

        gt_newlist.append(disp_s)
        disp_newlist.append(disp)
        
        nr_incons = np.count_nonzero(np.isnan(disp_s))
        nr_incons_tot = nr_incons_tot + nr_incons
        
        #disp_srl = RL_Check(dispRL, disp, 'MB')
        #gtrl_newlist.append(disp_srl)
        
        
        
        #writePFM(out_folder +  s_name + '%06d_e%06f.pfm' %(epoch,two_pe),disp.astype(np.float32))
        #writePFM(out_folder + s_name + '%06d_s.pfm' %epoch,disp_s) 
       # writePFM(out_folder + s_name + '%06d_rl.pfm' %epoch,dispRL.astype(np.float32))         

    
    avg_four_pe = avg_four_pe / len(left_list)
    avg_two_pe = avg_two_pe / len(left_list)
    avg_one_pe = avg_one_pe / len(left_list)
    avg_pf_pe = avg_pf_pe / len(left_list)

    print("4-PE: {}".format(avg_four_pe))
    print("2-PE: {}".format(avg_two_pe))
    print("1-PE: {}".format(avg_one_pe))
    print("0.5-PE: {}".format(avg_pf_pe))

    #if(avg_err > nr_incons_tot):
        #for j in range(0, len(left_list)):

    return gt_newlist, nr_incons_tot, disp_newlist

def RunEval(epoch, output_folder,filtered,lr_check,fill_incons,isTrain,dataset, save):
    nr = 0
    gt_newlist = []
    disp_newlist = []
    nr_incons_tot = 0.0
    
    
    avg_five_pe = 0.0
    avg_four_pe = 0.0
    avg_three_pe = 0.0
    avg_two_pe = 0.0
    avg_one_pe = 0.0
    avg_pf_pe = 0.0      
    
    #sanity-check: if fill_incons then also lr_check
    if(fill_incons):
        lr_check = True
    
    #samples problem!!!! Fix that!!!
    for i in range(0, len(left_list)):

        max_disp = max_disp_list[i]
        s_name = s_name_list[i]
        
        gt = gt_list[i]
        cost_volLT, cost_volRT = createCostVolAllTogetherSimB(left_list[i],right_list[i],max_disp)
        
        disp_L = None
        disp_R = None
        disp_s = None
        
        if(filtered):    
            cost_vol_filteredn = filterCostVolBilatpyt(cost_volLT,left_list[i])
            cost_vol_filteredn = np.squeeze(cost_vol_filteredn.cpu().data.numpy())                
            disp_L = np.argmax(cost_vol_filteredn, axis=0)
            
            if(lr_check):
                cost_vol_RL_fn = filterCostVolBilatpyt(cost_volRT,right_list[i])
                cost_vol_RL_fn = np.squeeze(cost_vol_RL_fn.cpu().data.numpy())

                disp_R = np.argmax(cost_vol_RL_fn, axis=0)  
                disp_s = LR_Check(disp_L.astype(np.float32), disp_R.astype(np.float32), dataset)
                
                if(fill_incons):
                    disp_s_arr = np.array(disp_s)
                    im_disp = Image.fromarray(disp_s_arr) 
                    im_disp = np.dstack((im_disp, im_disp, im_disp)).astype(np.uint8)    

                    h,w = disp_s.shape

                    shifted = cv2.pyrMeanShiftFiltering(im_disp, 7, 7)

                    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 1,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    disp_filled  = FillIncons(thresh, disp_s_arr)
                    
                    disp_L = np.array(disp_L)
                    disp_filled = np.array(disp_filled)  
                    
                
        else:
            
            cost_vol = np.squeeze(cost_volLT.cpu().data.numpy())
            disp = np.argmax(cost_vol, axis=0)    
            
            if(lr_check):
                
                disp_L = np.argmax(cost_volLT.cpu().data.numpy(), axis=0)
                disp_R = np.argmax(cost_volRT.cpu().data.numpy(), axis=0)
                
                disp_s = LR_Check(disp_L.astype(np.float32), disp_R.astype(np.float32), dataset)
                
                if(fill_incons):
                    disp_s_arr = np.array(disp_s)
                    im_disp = Image.fromarray(disp_s_arr) 
                    im_disp = np.dstack((im_disp, im_disp, im_disp)).astype(np.uint8)    

                    h,w = disp_s.shape

                    shifted = cv2.pyrMeanShiftFiltering(im_disp, 7, 7)

                    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 1,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                    disp_filled  = FillIncons(thresh, disp_s_arr)
                    
                    disp_L = np.array(disp_L)
                    disp_filled = np.array(disp_filled)      
        
        
        if(isTrain):
            if(fill_incons):
                five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp_filled, gt)
                avg_five_pe = avg_five_pe + five_pe
                avg_four_pe = avg_four_pe +  four_pe
                avg_three_pe = avg_three_pe + three_pe
                avg_two_pe = avg_two_pe + two_pe
                avg_one_pe = avg_one_pe + one_pe
                avg_pf_pe = avg_pf_pe + pf_pe  
                if(save):
                    writePFM(output_folder +  s_name + '_er%04f_ep%06d.pfm' %(two_pe,epoch),disp_filled.astype(np.float32)) 
                    writePFM(output_folder + s_name + '%06d_s.pfm' %epoch,disp_s) 
                    writePFM(output_folder + s_name + '%06d_R.pfm' %epoch,disp_R.astype(np.float32)) 

            else:
                #does not matter, variables are the same!!!!...fix
                if(filtered):
                    five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp_L, gt)
                    
                    avg_five_pe = avg_five_pe + five_pe
                    avg_four_pe = avg_four_pe +  four_pe
                    avg_three_pe = avg_three_pe + three_pe
                    avg_two_pe = avg_two_pe + two_pe
                    avg_one_pe = avg_one_pe + one_pe
                    avg_pf_pe = avg_pf_pe + pf_pe  
                    
                    if(save):
                        writePFM(output_folder + s_name + '_er%04f_ep%06d.pfm' %(two_pe,epoch),disp_L.astype(np.float32))
                        if(lr_check):
                            writePFM(output_folder + s_name + '%06d_s.pfm' %epoch,disp_s) 
                            writePFM(output_folder + s_name + '%06d_R.pfm' %epoch,disp_R.astype(np.float32)) 
                else:
                    
                    five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp_L, gt)
                    
                    avg_five_pe = avg_five_pe + five_pe
                    avg_four_pe = avg_four_pe +  four_pe
                    avg_three_pe = avg_three_pe + three_pe
                    avg_two_pe = avg_two_pe + two_pe
                    avg_one_pe = avg_one_pe + one_pe
                    avg_pf_pe = avg_pf_pe + pf_pe  

                    if(save):
                        writePFM(output_folder + s_name + '_er%04f_ep%06d.pfm' %(two_pe,epoch),disp_L.astype(np.float32))
                        if(lr_check):
                            writePFM(output_folder + s_name + '%06d_s.pfm' %epoch,disp_s) 
                            writePFM(output_folder + s_name + '%06d_R.pfm' %epoch,disp_R.astype(np.float32)) 
                    
        
        #needed for training!
            gt_newlist.append(disp_s)
            disp_newlist.append(disp_L)

            nr_incons = np.count_nonzero(np.isnan(disp_s))
            nr_incons_tot = nr_incons_tot + nr_incons        
        
        #redo this??
        

    if(isTrain):
    
        avg_four_pe = avg_four_pe / len(left_list)
        avg_two_pe = avg_two_pe / len(left_list)
        avg_one_pe = avg_one_pe / len(left_list)
        avg_pf_pe = avg_pf_pe / len(left_list)

        print("4-PE: {}".format(avg_four_pe))
        print("2-PE: {}".format(avg_two_pe))
        print("1-PE: {}".format(avg_one_pe))
        print("0.5-PE: {}".format(avg_pf_pe))        
        

    return gt_newlist, nr_incons_tot, disp_newlist 

def loadJack():

    #left_list = glob.glob('/home/dominik/tests2p/trainNW/SAda-Net/Input/im_right_tif/*.tif')
    left_list = glob.glob('/media/HDD/s2pOutput/JackOnlyBranchSlightlyTrained/im_left_tif/*.tif')
    left_list = sorted(left_list)
    
    #right_list = glob.glob('/home/dominik/tests2p/trainNW/SAda-Net/Input/im_left_tif/*.tif')
    right_list = glob.glob('/media/HDD/s2pOutput/JackOnlyBranchSlightlyTrained/im_right_tif/*.tif')
    right_list = sorted(right_list)
    
    #disprange_list = glob.glob('/home/dominik/tests2p/trainNW/SAda-Net/Input/disp_minmax/*.txt')
    disprange_list = glob.glob('/media/HDD/s2pOutput/JackOnlyBranchSlightlyTrained/disp_minmax/*.txt')
    
    disprange_list = sorted(disprange_list)
    
    left_list_train = []
    right_list_train = []
    maxdisp_list = []
    s_name_list = []

    
    for i in range(0,len(left_list)):
        
        im_left_tif = gdal.Open(left_list[i])
        im_left_band = im_left_tif.GetRasterBand(1)
        im_left_arr = im_left_band.ReadAsArray()
        im_left_arr[np.where(np.isnan(im_left_arr))] = 0.0
        
        left_list_train.append(im_left_arr)
        
        im_right_tif = gdal.Open(right_list[i])
        im_right_band = im_right_tif.GetRasterBand(1)
        im_right_arr = im_right_band.ReadAsArray()
        im_right_arr[np.where(np.isnan(im_right_arr))] = 0.0
        
        right_list_train.append(im_right_arr)
        
        f = open(disprange_list[i],'r')
        calib = f.read()
        max_disp = float(calib.split('\n')[1]) 
        
        min_disp = float(calib.split('\n')[0])
        
        max_disp = np.abs(int(np.ceil(max_disp)))
        
        min_disp = np.abs(int(np.ceil(min_disp)))
        
        disp = max_disp + min_disp + 1
        maxdisp_list.append(disp)
        
        s_name_list.append(left_list[i].split('/')[-1].split('.')[0])

        
    return left_list_train, right_list_train, maxdisp_list, s_name_list

left_list, right_list, maxdisp_list, s_name_list = loadJack()


def RunPred(epoch, save, w_filter):
    
    disp_s_list = []
    disp_L_list = []

    nr_incons_tot = 0.0
    
    for i in range(len(left_list)):
        
        left = left_list[i]
        right = right_list[i]
        maxdisp = maxdisp_list[i]
        h,w = left.shape
        
        
        
        cost_volL, cost_volR = createCostVolAllTogether(left, right, maxdisp)
        
        if(w_filter):
            
            cost_volRfiltered = filterCostVolBilatpyt(cost_volR,right)
            cost_volRfiltered = np.squeeze(cost_volRfiltered.cpu().data.numpy())                

            cost_volLfiltered = filterCostVolBilatpyt(cost_volL,left)
            cost_volLfiltered = np.squeeze(cost_volLfiltered.cpu().data.numpy()) 

            dispL = np.argmax(cost_volLfiltered, axis=0) 
            dispR = np.argmax(cost_volRfiltered, axis=0) 
            
        else:
            dispL = np.argmax(cost_volL.cpu().data.numpy(), axis=0) 
            dispR = np.argmax(cost_volR.cpu().data.numpy(), axis=0) 
            
        
        disp_s = RL_Check(dispL.astype(np.float32), dispR.astype(np.float32))
        
       

        if(save == True):
            
            writePFM(out_folder + '%05d_%03d'%(epoch,i) + '_s.pfm',disp_s.astype(np.float32)) 

            

        nr_incons = np.count_nonzero(np.isnan(disp_s))
        nr_incons_tot = nr_incons_tot + nr_incons
        
        disp_s_list.append(disp_s)
        disp_L_list.append(dispL)
            
    return disp_s_list, disp_L_list, nr_incons_tot


def createCostVolAllTogether(left_im,right_im,max_disp):    
    
    a_h, a_w = left_im.shape
    
    #left_im = stretch(left_im)
    #right_im = stretch(right_im)
    
    left_im = left_im[np.newaxis,...]
    right_im = right_im[np.newaxis,...]
    
    left_im = np.reshape(left_im, [1,1,a_h,a_w])
    right_im = np.reshape(right_im, [1,1,a_h,a_w])
    
    with torch.no_grad():

        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        #no image norm?? try it!
        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_volL = np.zeros((max_disp+1,a_h,a_w))
        cost_volLT = Variable(Tensor(cost_volL))   
    
        cost_volR = np.zeros((max_disp+1,a_h,a_w))
        cost_volRT = Variable(Tensor(cost_volR))   
    
        #0 => max_disp => one less disp!
        for disp in range(0,max_disp+1):
            
            if(disp == 0):
                #sim_score_l = cos(left_feat, right_feat)
                #sim_score_r = cos(left_feat, right_feat)

                sim_score_l = simB(torch.cat((left_feat, right_feat),dim=1))
                sim_score_r = simB(torch.cat((left_feat, right_feat),dim=1))

                cost_volRT[disp,:,:] = torch.squeeze(sim_score_l) 
                cost_volLT[disp,:,:] = torch.squeeze(sim_score_r) 

            else:
                
                left_shifted = torch.roll(left_feat, disp, dims = 3) 
                left_shifted[:,:,:,w-disp:w] = 0
                
                #sim_score_right = cos(left_shifted, right_feat)
                sim_score_right = simB(torch.cat((left_shifted, right_feat),dim=1))
                
                cost_volRT[disp,:,:] = torch.squeeze(sim_score_right) 
                
    
                right_shifted = torch.roll(right_feat, -disp, dims = 3) 
                right_shifted[:,:,:,0:disp] = 0
            
                #sim_score_left = cos(left_feat, right_shifted)
                sim_score_left = simB(torch.cat((left_feat, right_shifted),dim=1))

                cost_volLT[disp,:,:] = torch.squeeze(sim_score_left) 
                

    return cost_volLT, cost_volRT


def createCostVolAllTogetherBranch(left_im,right_im,max_disp):    
    
    a_h, a_w = left_im.shape
    
    #left_im = stretch(left_im)
    #right_im = stretch(right_im)
    
    left_im = left_im[np.newaxis,...]
    right_im = right_im[np.newaxis,...]
    
    left_im = np.reshape(left_im, [1,1,a_h,a_w])
    right_im = np.reshape(right_im, [1,1,a_h,a_w])
    
    with torch.no_grad():

        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        #no image norm?? try it!
        left_imT = (left_imT-torch.mean(left_imT))/torch.std(left_imT)
        right_imT = (right_imT-torch.mean(right_imT))/torch.std(right_imT)

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_volL = np.zeros((max_disp+1,a_h,a_w))
        cost_volLT = Variable(Tensor(cost_volL))   
    
        cost_volR = np.zeros((max_disp+1,a_h,a_w))
        cost_volRT = Variable(Tensor(cost_volR))   
    
        #0 => max_disp => one less disp!
        for disp in range(0,max_disp+1):
            
            if(disp == 0):
                sim_score_l = cos(left_feat, right_feat)
                sim_score_r = cos(left_feat, right_feat)

                cost_volRT[disp,:,:] = torch.squeeze(sim_score_l) 
                cost_volLT[disp,:,:] = torch.squeeze(sim_score_r) 

            else:
                
                left_shifted = torch.roll(left_feat, disp, dims = 3) 
                left_shifted[:,:,:,w-disp:w] = 0
                
                sim_score_right = cos(left_shifted, right_feat)
                cost_volRT[disp,:,:] = torch.squeeze(sim_score_right) 
                
    
                right_shifted = torch.roll(right_feat, -disp, dims = 3) 
                right_shifted[:,:,:,0:disp] = 0
            
                sim_score_left = cos(left_feat, right_shifted)
                cost_volLT[disp,:,:] = torch.squeeze(sim_score_left) 
                

    return cost_volLT, cost_volRT

def getGTStart(epoch, save, w_filter):
    
    disp_s_list = []
    disp_L_list = []

    nr_incons_tot = 0.0
    
    for i in range(len(left_list)):
        
        left = left_list[i]
        right = right_list[i]
        maxdisp = maxdisp_list[i]
        h,w = left.shape
        
        
        
        cost_volL, cost_volR = createCostVolAllTogetherBranch(left, right, maxdisp)
        
        if(w_filter):
            
            cost_volRfiltered = filterCostVolBilatpyt(cost_volR,right)
            cost_volRfiltered = np.squeeze(cost_volRfiltered.cpu().data.numpy())                

            cost_volLfiltered = filterCostVolBilatpyt(cost_volL,left)
            cost_volLfiltered = np.squeeze(cost_volLfiltered.cpu().data.numpy()) 

            dispL = np.argmax(cost_volLfiltered, axis=0) 
            dispR = np.argmax(cost_volRfiltered, axis=0) 
            
        else:
            dispL = np.argmax(cost_volL.cpu().data.numpy(), axis=0) 
            dispR = np.argmax(cost_volR.cpu().data.numpy(), axis=0) 
            
        
        disp_s = RL_Check(dispL.astype(np.float32), dispR.astype(np.float32))
        
       

        if(save == True):
            
            writePFM(out_folder + '%05d_%03d'%(epoch,i) + '_s.pfm',disp_s.astype(np.float32)) 

            

        nr_incons = np.count_nonzero(np.isnan(disp_s))
        nr_incons_tot = nr_incons_tot + nr_incons
        
        disp_s_list.append(disp_s)
        disp_L_list.append(dispL)
            
    return disp_s_list, disp_L_list, nr_incons_tot

gt_newlist, disp_final, nr_incons_tot_start = getGTStart(-1, True, True)

lr = 0.00006
batch_size = 400
nr_epochs = 20000000
save_weights = 1000


params = list(branch.parameters()) + list(simB.parameters()) #+ list(simB.parameters())

optimizer_G = optim.Adam(params, lr)

best_err = 10000000000000

early_stopping_count = 0

incons_list = []


i = 0

while(True):


    epoch_loss = 0.0

    #reset gradients
    optimizer_G.zero_grad()

    batch_xl, batch_xr_pos, batch_xr_neg = getBatch(gt_newlist)
    bs, c, h, w = batch_xl.shape

    xl = Variable(Tensor(batch_xl))

    xr_pos = Variable(Tensor(batch_xr_pos))
    xr_neg = Variable(Tensor(batch_xr_neg))


    xl = (xl-torch.mean(xl))/torch.std(xl)
    xr_pos = (xr_pos-torch.mean(xr_pos))/torch.std(xr_pos)      
    xr_neg = (xr_neg-torch.mean(xr_neg))/torch.std(xr_neg) 

    left_out = branch(xl)
    right_pos_out = branch(xr_pos)
    right_neg_out = branch(xr_neg)


#can this even work??? we dont train simb!

    sp = simB(torch.cat((left_out, right_pos_out),dim=1))
    sn = simB(torch.cat((left_out, right_neg_out),dim=1))   

    #sp = cos(left_out, right_pos_out)
    #sn = cos(left_out, right_neg_out)   

    batch_loss = my_hinge_loss(sp, sn)
    batch_loss = batch_loss.mean()



    batch_loss.backward()
    optimizer_G.step()

    if(save_weights >= 1000):
        save_weights = 100

    if(i % save_weights == 0):
        if(i > 0):

            print("EPOCH: {} loss: {}".format(i,batch_loss))

            #epoch
            #save = False
            #filter = True
            new_disps, disp_final, nr_incons_tot = RunPred(i, True, True)

            incons_list.append(nr_incons_tot)

            if(nr_incons_tot < nr_incons_tot_start):

                gt_newlist = new_disps
                nr_incons_tot_start = nr_incons_tot

            if(nr_incons_tot < best_err):

                early_stopping_count = 0
                print("Improved")
                print("Incons: {}".format(nr_incons_tot))

                torch.save(simB.state_dict(), save_folder_simb + model_name + '_best%04i' %(i) + 'e%04f' %(nr_incons_tot)) 
                torch.save(branch.state_dict(), save_folder_branch + model_name + '_best%04i' %(i) + 'e%04f' %(nr_incons_tot)) 

                #best_err = nr_incons_tot
                best_err = nr_incons_tot

            else:

                print("got worse")
                print("Incons: {}".format(nr_incons_tot))
                early_stopping_count = early_stopping_count + 1


            if(early_stopping_count >= 20):

                print("Early-stop Epoch: {}".format(i))
                print('----------------------------------')

                break

    i = i + 1            





