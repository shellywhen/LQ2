#!/usr/bin/env python
# -*- coding: utf-8 -*-
class ModelArg():
    """A class to mimic the command line parser.
    ----------------EXPERIMENT------------------
    :name: an alias to the experiment
    :model_save_to: path of the saved file, including log and parameters
    ------------------DATASET------------------
    :val_split_ratio: percentage of the validation data split from the dataset
    :param_path: path to the parameter list file  
    :label_path: path to the mturk result file
    :extreme_path: path to the stastistics of the parameter for normalization
    :fields: identifiers of the layout parameters
    -----------------TRAINING------------------
    :lr: learning rate
    :epoch: epoch
    :batch_size: the size of a mini-batch
    :seed: ranseed for data split
    -----------------HYBRID MODEL--------------
    :linear_list: the middle layers of the final rating MLP
    """
    def __init__(self, 
                 label_path='dataset/exp1/turk_results.csv',
                 param_path='dataset/exp1/parameters.csv',
                 extreme_path='dataset/exp1/extremes.csv',
                 val_split_ratio=0.2,
                 seed=5783287,
                 batch_size=128,
                 epoch=200,
                 lr=0.5,
                 linear_list=[64, 32, 16],
                 dropout_rate=0.2,
#                  model_save_to='checkpoint/',
                 fields=['width', 'nbar', 'bandwidth'],
                 name='model',
                ):
 
        self.label_path = label_path
        self.param_path = param_path
        self.extreme_path = extreme_path
        self.batch_size = batch_size
        self.val_split_ratio = val_split_ratio
        self.epoch = epoch
        self.lr = lr
        self.seed = seed
        self.fields = fields
        self.model_save_to = 'checkpoint/'
        self.linear_list = linear_list    
        self.dropout_rate = dropout_rate
        self.name = name
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
        
       
      