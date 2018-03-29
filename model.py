#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:34:20 2018

@author: lcy
"""

from torchvision import models

vgg = models.vgg16(pretrained=True)
