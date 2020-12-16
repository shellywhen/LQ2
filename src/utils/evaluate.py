#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

def loss_func(y, c):
    """Test for a suitable loss function for the training process
      :scores: ((n), (n)) tensor of the score of good and bad samples.
    """
    s0, s1 = y
    winningCount, loserCount = c
    m = Variable(torch.Tensor([0.12]).double()).expand(s1.size()).cuda()
    c0 = Variable(torch.Tensor([0]).double()).expand(s1.size()).cuda()
    eps = torch.max(-s0 + s1 + m, c0)
    res = torch.sum(eps)
    eps =  torch.exp((s0-s1-1)) + torch.exp((s1 - s0)) 
    return res


def correct(y, device=torch.device('cuda:0')):
    """Evaluate the model on train data (with a particular order)
    NOTE THAT ONLY A DELTA OF 0 IS CONSIDERED VALID.
    :y: tuple ([b], [b]) of the score of the good and bad samples
    :device: device
    """
    y = torch.stack((y[0],y[1])).to(device)
    judge_mask = y[0]-y[1] > 0
    correct = torch.sum(judge_mask)
#     winners = torch.argmax(y, dim=0).to(device)
#     gt = torch.zeros(y.shape[1], dtype=torch.int64).to(device)
#     compare = torch.eq(winners, gt).to(device)
#     correct = torch.sum(compare).to(device)
    return correct
