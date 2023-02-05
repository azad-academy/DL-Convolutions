'''
Author: J. Rafid Siddiqui
Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================



import numpy as np
from PIL import Image, ImageDraw
import math
class Convolution2D:

    def __init__(self):

        self.img = None
        self.filter = np.array([ [[-1,-2,-1],[0,0,0],[1,2,1]], [[-1,0,1],[-2,0,2],[-1,0,1]], [[0,-1,-2],[1,0,-1],[2,1,0]], [[-2,-1,0],[-1,0,1],[0,1,2]] ])  #4x (3x3) Kernels
        self.img_arr = None
        self.conv_imgs = None
        self.fig = None
        self.patch_size = 125
        self.cur_pos = np.array([0,0],dtype=int)
        self.cur_patch = np.zeros([self.patch_size,self.patch_size])
        self.img_axes = None
        self.conv_axes = None
        self.feature_axes = None
        self.features = np.zeros((12,self.patch_size,self.patch_size))
        self.anim = None
        

    def convolve(self,img,K):
        
        C_img = np.zeros(img.shape)
        h = img.shape[0]
        w = img.shape[1]
        ksize = K.shape[0]

        pad_w = math.ceil(ksize/2) 
        pad_h = math.ceil(ksize/2) 
        padded_img = np.pad(img, pad_width=[(pad_w, pad_h),(pad_w, pad_h)], mode='constant', constant_values=(255,255))
        
        for j in range(w):
            for i in range(h):
                window = padded_img[i:i+ksize, j:j+ksize]
                if(not window.shape[0]):
                    break
                C_img[i,j] = (K * window).sum()                 
                
        return C_img.astype(int)

    def convolve_img(self,img,K):

        C_img = np.zeros(img.shape,dtype=int)
        
        C_img[:,:,0] = self.convolve(img[:,:,0],K) 
        C_img[:,:,1] = self.convolve(img[:,:,1],K) 
        C_img[:,:,2] = self.convolve(img[:,:,2],K)
                
        return C_img