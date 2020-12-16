#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data Preprocessing and loader for pairwise data from the dataset.
Items concern the original parameter from the Vega-Lite specifications.
"""
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import torchvision
import torch
import json
import random
from collections import OrderedDict
from utils.model_arg import ModelArg
    
class Loader:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.extremes =  pd.read_csv(args.extreme_path, encoding='utf-8').set_index('field').to_dict('index')
        
    def minmax(self, value, cat):
        return (value - self.extremes[cat]['minvalue'])/(self.extremes[cat]['maxvalue']-self.extremes[cat]['minvalue'])
    
    def deminmax(self, value, cat):
        return value * (self.extremes[cat]['maxvalue']-self.extremes[cat]['minvalue']) + self.extremes[cat]['minvalue']
   
    def batch_denormalize(self, df):
        for field in self.args.fields:
            df[field] = df[field].apply(lambda x: self.deminmax(x, field))
        return df
    
    def load_parameter(self):
        df = pd.read_csv(self.args.param_path, encoding='utf-8')
        df['name'] = df['name'].apply(lambda x: x.split('/')[-1].replace('.png', ''))
        for field in self.args.fields:
            df[field] = df[field].apply(lambda x: self.minmax(x, field))
        param_dict = df.to_dict('records')
        final = {}
        for d in param_dict:
            final[d['name']] = torch.FloatTensor([d[field] for field in self.args.fields]).to(self.device)
        return final
    
    def load_dataset(self):
        """Load, normalize, and split the dataset for training and evaluating.
        Returns the training loader and the validating loader.
        """
        args = self.args
        device = self.device
        random.seed(args.seed)
        df = pd.read_csv(args.label_path, encoding='utf-8')[['good', 'bad', 'goodCount', 'badCount']].drop_duplicates()
        label_data = df.values
        param_dict = self.load_parameter()
        data_fold = DataPair(label_data, param_dict, device)
        data_size = len(data_fold)
        indices = list(range(data_size))
        split = int(np.floor(args.val_split_ratio * data_size))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[0:split] 
        # print(f'size={data_size} split={split} train_indices={len(train_indices)} val_indices={len(val_indices)}')                   
        ### Create data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = data.DataLoader(dataset=data_fold, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
        val_loader = data.DataLoader(dataset=data_fold, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
        return train_loader, val_loader
    
    def load_model(self, folder, model_name='final'):
        """Load the trained model.
        :folder: location of the model
        :model_name: name of the model
        """
        model_path = f'{self.args.model_save_to}{folder}/{model_name}.pt'
        model = torch.load(model_path)
        model.eval()
        model.to(self.device)
        return model


class DataPair(Dataset):
    """
    The dataset for pairwise features including scaled image, parameter, and bar sequence.
    """
    def __init__(self, turk_dataset, params, device):
        """ Initialize the PairData dataset.
            :turk_dataset: a list of (#img0, #img1, #winner, #loser, cidx)
            :device: device assigned in Pytorch
        """
        self.turk_dataset = turk_dataset
        self.params = params
        self.device = device
    def __getitem__(self, idx):
        """Returns a pair of valid triplet for training
           --------- NOTE: #0 is always the winner ----------
        """
        # winner_id = int(self.turk_dataset[idx][4])
        img_url_0 = self.turk_dataset[idx][0].split('/')
        img_url_1 = self.turk_dataset[idx][1].split('/')
        img_name_0 = img_url_0[-1].replace('.png', '')
        img_name_1 = img_url_1[-1].replace('.png', '')
        param_0 = self.params[img_name_0]
        param_1 = self.params[img_name_1]
        return  (param_0, param_1), (img_name_0, img_name_1), self.turk_dataset[idx][2], self.turk_dataset[idx][3]

    def __len__(self):
        """Returns the length of the training data from crowdsourcing.
        """
        return len(self.turk_dataset)
