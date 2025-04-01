# -*- coding: UTF-8 -*-

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import os
import scipy.io as scio
from skimage.io import imread, imsave
import numpy as np
import torch.nn.functional as F
import cv2
from os.path import exists

class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std
        image = (image-image.min())/(image.max()-image.min())
        if mask is None:
            return image
        return image, mask/255.0

class RandomCrop(object):
    def __call__(self, image, mask=None):
        H,W   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1].copy()
            return image[:,:,::-1].copy(), mask[:,:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask

def Resize(image, mask,H,W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask  = cv2.resize( mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        return image, mask
    else:
        return image

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)

        return image, mask

def _resize_image(image, target):
   return cv2.resize(image, dsize=(target[0], target[1]), interpolation=cv2.INTER_LINEAR)


label_ls = ['reflex_overall',  'corneal scar',  'corneal thinning','pupil','stromal infiltrate','surrounding inflammation','hypopyon']

class MyDataset(data.Dataset):# 
    def __init__(self, args, train_list, mode,root):  
        self.args = args
        img_ls, mask_ls, name_ls, limbus_ls = [], [],[], []
        for pat in train_list:
            img_path = root+pat
            # print(img_path)
            gt_l = glob.glob(img_path+'/gt/*/')
            # print(img_ls)
            for gt in gt_l:
                # print(img)
                name = gt.split('/')[-2]
                

                img = img_path+'/img/'+name+'.jpg'
                limbus = '/home/ziyun/Desktop/project/MK project/segmentation_code/Bicon/Oct/W/35_0/output/' +name+'.png'
                if os.path.exists(img):
                    img_ls.append(img)
                    mask_ls.append(gt)
                    name_ls.append(name)
                    limbus_ls.append(limbus)
        
        self.mode = mode

        self.limbus_ls = limbus_ls
        self.name_ls = name_ls
        self.img_ls = img_ls

        self.mask_ls = mask_ls

        self.normalize  = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        self.totensor   = ToTensor()

    def __getitem__(self, index):
        # print(img_ls[index])
        
        limbus_mask = cv2.imread(self.limbus_ls[index]).astype(np.float32)
        img  = cv2.imread(self.img_ls[index]).astype(np.float32)

        limbus_mask = limbus_mask/255.0
        if np.max(limbus_mask) != 0:
            x_scale = img.shape[0]/512
            y_scale = img.shape[1]/768
            # print(x_scale, y_scale)
            coord = np.where(limbus_mask==1)
            # print(coord)
            x_min = max(np.min(coord[0])-10 , 0)
            x_max = (np.max(coord[0])+10 ) 
            y_min = max(np.min(coord[1])-10 , 0) 
            y_max = (np.max(coord[1]) +10 )
            
            valid_limbus = limbus_mask[x_min:x_max,y_min:y_max]
            # imsave('valid_limbus.png',limbus_mask[x_min:x_max,y_min:y_max])
            
            x_min = int(x_min*x_scale)
            x_max = int(x_max *x_scale)
            y_min = int(y_min* y_scale)
            y_max = int(y_max* y_scale)

        else:
            x_min = 0
            x_max = -1
            y_min = 0
            y_max = -1

        img = img[x_min:x_max,y_min:y_max,:]

        img = cv2.resize(img,dsize=(self.args.resize[1], self.args.resize[0]))

        img = img[:, :, [2, 1, 0]]


        # print(self.mask_ls[index]+'/'+str(7)+'.png')
        # mask = cv2.imread(self.mask_ls[index]+'/1.png',0).astype(np.float32) if exists(self.mask_ls[index]+'/1.png') else np.zeros([512,512])
        mask = [cv2.resize(cv2.imread(self.mask_ls[index]+'/'+label_ls[i]+'.png',0).astype(np.float32)[x_min:x_max,y_min:y_max],dsize=(self.args.resize[1], self.args.resize[0])) if exists(self.mask_ls[index]+'/'+label_ls[i]+'.png') else np.zeros(self.args.resize) for i in range(len(label_ls))]
        

        mask = np.array(mask)
        
        img = np.transpose(img,(2,0,1))
        # print(img.shape)
        # print(img.shape)
        img,mask = self.normalize(img,mask)
        # limbus_mask = limbus_mask/255.0
        # print(mask.shape,img.shape)
        if self.mode == 'train':
            img, mask = self.randomflip(img, mask)
            # limbus_mask = self.randomflip(limbus_mask)


        
        # limbus_mask  = torch.from_numpy(limbus_mask)

        mask = torch.where(mask>0.5,1,0)

        return img,mask#,limbus_mask.unsqueeze(0)
    



    def __len__(self):  
        return len(self.img_ls)


def connectivity_matrix(multimask, class_num):
    # print(multimask.shape)
    # multimask = multimask.squeeze()
    [_,rows, cols] = multimask.shape
    batch = 1
    conn = torch.zeros([batch,class_num*8,rows, cols])


    for i in range(class_num):
        mask = multimask[i,:,:]

        up = torch.zeros([batch,rows, cols])#move the orignal mask to up
        down = torch.zeros([batch,rows, cols])
        left = torch.zeros([batch,rows, cols])
        right = torch.zeros([batch,rows, cols])
        up_left = torch.zeros([batch,rows, cols])
        up_right = torch.zeros([batch,rows, cols])
        down_left = torch.zeros([batch,rows, cols])
        down_right = torch.zeros([batch,rows, cols])


        up[:,:rows-1, :] = mask[1:rows,:]
        down[:,1:rows,:] = mask[0:rows-1,:]
        left[:,:,:cols-1] = mask[:,1:cols]
        right[:,:,1:cols] = mask[:,:cols-1]
        up_left[:,0:rows-1,0:cols-1] = mask[1:rows,1:cols]
        up_right[:,0:rows-1,1:cols] = mask[1:rows,0:cols-1]
        down_left[:,1:rows,0:cols-1] = mask[0:rows-1,1:cols]
        down_right[:,1:rows,1:cols] = mask[0:rows-1,0:cols-1]


        conn[:,(i*8)+0,:,:] = mask*down_right
        conn[:,(i*8)+1,:,:] = mask*down
        conn[:,(i*8)+2,:,:] = mask*down_left
        conn[:,(i*8)+3,:,:] = mask*right
        conn[:,(i*8)+4,:,:] = mask*left
        conn[:,(i*8)+5,:,:] = mask*up_right
        conn[:,(i*8)+6,:,:] = mask*up
        conn[:,(i*8)+7,:,:] = mask*up_left

    conn = conn.float()
    conn = conn.squeeze()
    # print(conn.shape)
    return conn



def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def check_label(mask):
    label = np.array([1,0,0,0])
    # print(mask.shape)
    # print(mask[1,:,:].max())
    if mask[1,:,:].max()!=0:
        label[1]=1

    if mask[2,:,:].max()!=0:
        label[2]=1

    if mask[3,:,:].max()!=0:
        label[3]=1

    return label

# def thres_multilabel(mask):
#     mask[np.where(mask<0.5)]=0
#     mask[np.where((mask<1.5) & (mask>=0.5))]=1
#     mask[np.where((mask<2.5) & (mask>=1.5))]=2
#     mask[np.where(mask>2.5)]=3

#     return mask
