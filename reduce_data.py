#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:19:48 2018

@author: xinjie
"""

import pickle as cPickle


target_params = cPickle.load(open('save/reduced_target_params.pkl','rb'), encoding='iso-8859-1')

target_params[0] = target_params[0][0:20, :]
target_params[13] = target_params[13][:, 0:20]
target_params[14] = target_params[14][0:20]


output = open('/Users/xinjie/Documents/GitHub/SeqGAN1/save/reduced_target_params.pkl', 'wb')
cPickle.dump(target_params, output, protocol=2)

target_params1 = cPickle.load(open('save/reduced_target_params.pkl','rb'), encoding='iso-8859-1')