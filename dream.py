#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:34:06 2018

@author: lcy
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd

from util import showMaxID,showImg,saveImg,tmpDirPath

def default_diff(x,guide = None):
    return x.data

def objective_guide_diff(dst, guide_features):
    x = dst.data[0].cpu().numpy().copy()
    y = guide_features.data[0].cpu().numpy()
    ch, w, h = x.shape
    x = x.reshape(ch,-1)
    print(x.shape)
    y = y.reshape(ch,-1)
    print(y.shape)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    result = y[:,A.argmax(1)] # select ones that match best
    result = torch.Tensor(np.array([result.reshape(ch, w, h)], dtype=np.float))
    return result


def dream_step(img, model,layerNum=6, control_features = None,
               learning_rate = 2e-2, max_jitter = 32, num_iterations = 20, **kwargs):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
    difffunc = default_diff if control_features is None else objective_guide_diff
    
    
    modulelist = list(model.modules())
    
    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        model.zero_grad()
        if isinstance(img, np.ndarray):
            img_variable = Variable(torch.from_numpy(img), requires_grad=True)
        else:
            img_variable = Variable(img, requires_grad=True)
#        act_value = model.forward(img_variable, end_layer)
        x = img_variable
        for j in range(1, layerNum):
            x = modulelist[j+1](x)
        
        diff_out = difffunc(x, control_features)#distance(act_value, guide_features)
        x.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std,
                                  (1 - mean) / std)
        
#        showImg(img[0])
    return img


def dream(model, img, layerNum=4, octave_n=6, octave_scale=1.1,savename = 'tmp.jpg',
          savedir = tmpDirPath, control_tensor = None,
          learning_rate = 2e-2, max_jitter = 32, num_iterations = 20, **kwargs):
    
    if control_tensor is not None:
        modulelist = list(model.modules())
        x = Variable(control_tensor, requires_grad = False)
        for j in range(1, layerNum):
            x = modulelist[j+1](x)
        control_features = x  
    else:
        control_features = None
        
    octaves = [img]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], 
                               (1, 1, 1.0 / octave_scale, 1.0 / octave_scale),
                               order=1)
                        )

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(
                detail, (1, 1, h / h1,  w / w1), order=1)

        input_oct = octave_base + detail
        print(input_oct.shape)
        out = dream_step(input_oct, model, layerNum, control_features, **kwargs)
        saveImg(out[0], savename, savedir)
#        showImg(out[0])
        detail = out - octave_base
