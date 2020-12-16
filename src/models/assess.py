#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORE MODEL handling all kinds of input.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Assess(nn.Module):
    def __init__(self, 
                 device,
                 param_dim = 6,
                 linear_list = [64, 32, 16],
                 dropout_rate=0.2
                 ):
        """Initialization of the multi-input model.
        :device: cpu/gpu
        :param_dim: dimension of vis parameters
        :linearlist: the internal hidden MLP layers from feature to the score
        """
        super(Assess, self).__init__()
        self.param_dim = param_dim
        layers  = []
        linear_list.insert(0, param_dim)
        for i in range(len(linear_list)-1):
            layers.append(nn.Linear(linear_list[i], linear_list[i+1]))
            layers.append(nn.LeakyReLU()) 
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(linear_list[-1], 1))
        layers.append(nn.Sigmoid())    
        self.linear = nn.Sequential(*layers)
        self.linear.apply(init_weights)
        
    def assess(self, x, names):
        s = self.linear(x)
        s = s.squeeze()
        value = s.detach().cpu().numpy()
        out = list(zip(names, value))
        return out
        
    def path(self, param):
        f = param
        s = self.linear(f)
        s = s.squeeze()
        return s
        
    def forward(self, x):
        x1, x2 = x
        s1 = self.path(param=x1)
        s2 = self.path(param=x2)
        return s1, s2
        
        
        
        
        