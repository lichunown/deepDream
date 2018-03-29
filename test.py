#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:11:35 2018

@author: lcy
"""

import os
from model import vgg
from dream import dream
from util import readImg,imgDirPath,tmpDirPath

guideslist = [None, 'flower.jpg','input.png', 'kitten.jpg']

def findExist(g,layerNum):
    return os.path.exists(os.path.join(tmpDirPath,'{}[{}]_2.jpg'.format(str(g),layerNum)))


for g in guideslist:
    if g:
        guide = readImg(os.path.join(imgDirPath,g))
    else:
        guide = None
    for layerNum in range(2,33):
        try:
            if not findExist(g,layerNum):
                print('[Running] guide:{}     layerNum:{}'.format(g, layerNum))
                img = readImg()
                dream(vgg, img, layerNum = layerNum, 
                      octave_n = 6, octave_scale=1.1, savename= '{}[{}].jpg'.format(str(g).replace('.jpg',''),layerNum),
                      is_control = True if guide is not None else False, control_tensor = guide,
                      learning_rate = 2e-2, max_jitter = 32, num_iterations = 30)
                del img
        except Exception as e:
            print('error:',e)
