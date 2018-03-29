#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:33:53 2018

@author: lcy
"""


import argparse
from util import tmpDirPath

parser = argparse.ArgumentParser(description='Deep Dream')

parser.add_argument('imgpath', type=str, default = './imgs/sky.jpg',
                    help='the origin image path')

parser.add_argument('--layerNum', type=int, default = 20,
                    help='vgg layers (2-32)')

parser.add_argument('--octave_n', type=int, default = 6,
                    help='octave n')

parser.add_argument('--octave_scale', type=int, default=1.1,
                    help='octave scale')

parser.add_argument('--savename', type=str, default = 'tmp.jpg',
                    help='save img name (auto add nums)')

parser.add_argument('--savedir', type=str, default = tmpDirPath,
                    help='save imgs dir')

parser.add_argument('--guide', type=str, default = None,
                    help='guide img path')

parser.add_argument('--learning_rate', type=float, default = 2e-2,
                    help='learning rate')

parser.add_argument('--num_iterations', type=float, default = 20,
                    help='per octave train iter')


from model import vgg
from dream import dream
from util import readImg

print(parser)

if parser.guide:
    guide = readImg(parser.guide)
else:
    guide = None

dream(vgg, readImg(parser.imgpath), layerNum = parser.layerNum, 
      octave_n = parser.octave_n, octave_scale = parser.octave_scale, savename = parser.savename,
      savedir = parser.savedir, control_tensor = guide,
      learning_rate = parser.learning_rate, max_jitter = 32, num_iterations = parser.num_iterations)




