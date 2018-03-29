#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:35:00 2018

@author: lcy
"""
import os
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np


imgDirPath = './imgs'
tmpDirPath = './result'

_testImg = 'sky.jpg'

if not os.path.exists(tmpDirPath):
    os.makedirs(tmpDirPath)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    
_imgPath = os.path.join(imgDirPath,_testImg)


def showMaxID(fname = None, path = tmpDirPath):
    if fname is None:
        ids = [int(name.split('.')[-2].split('_')[-1]) for name in os.listdir(path)]
    else:
        ids = [int(name.split('.')[-2].split('_')[1]) if name.split('.')[-2].split('_')[0] == fname else -1 
               for name in os.listdir(path)]
    if not ids:
        return 0
    else:
        return max(ids)+1
    
    
def readImg(path = _imgPath,dirs = './'):
    img = Image.open(os.path.join(dirs, path))
#    img.thumbnail( (int(img.size[0]/4), int(img.size[1]/4)), Image.ANTIALIAS)
    img = img_transform(img).unsqueeze(0)
    return img

#def showImg(img):
#    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
#    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
#    if not isinstance(img, np.ndarray):
#        img = img.numpy()
#    img = np.clip(np.transpose(img, (1,2,0)) * std + mean,0,1)
#    return img

def change2img(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.numpy()
    img = np.clip(np.transpose(tensor, (1,2,0)) * std + mean, 0, 255)
    return img
    
def change2Tensor(img):
    return img_transform(img).unsqueeze(0)
    
def saveImg(img, name = 'tmp.jpg', dirs = tmpDirPath):
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    if not isinstance(img, np.ndarray):
        img = img.numpy()
    img = np.transpose(img, (1,2,0)) * std + mean
    j = Image.fromarray(np.uint8(np.clip(img*255, 0, 255)))
    name = '{}_{}.{}'.format(   name.split('.')[0],
                                showMaxID(name.split('.')[0],dirs),
                                name.split('.')[1]   )
    j.save(os.path.join(dirs,name))
    
