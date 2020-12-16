# Readme

This folder contains code for training scoring model from paired crowdsourced data.

## Overview

- baseline: code for generating empirical graphic features as a scoring baseline compared with our methods
- checkpoint: trained models with the training log
- models: the model architecture
- utils: utility functions
- script
  - trainModel.ipynb
  - visualizeModel.ipynb

## Model Training

Please refer to *trainModel.ipynb* for model training. After running corresponding blocks, you will find trained models in the `~/src/checkpoint/` folder with a folder named by the corresponding timestamp.

 In the script, you can adapt the `ModelArg` object to pass specifications. Here are some common fields.

- model-relevant
  - `linear_list`: the size of each hidden layers in the mlp
- training-relevant
  - `lr`: learning rate
  - `batch_size`: batch size
  - `epoch`: number of epoch
  - `seed`: the seed for random numbers
- dataset-relevant
  - `label_path`: a url to the user study dataset of paired comparison
  - `param_path`: a url to the chart layout parameter dataset
  - `extreme_path`: a url to the computed extremes of the corresponding dataset for normalization
  - `fields`: a list of the parameter type string appeared in the header of the parameter file
  - `val_split_ratio`: the ratio (0~1) of the size of validation dataset in the whole dataset

Note that if you want to to train on your own data, you may need to follow the format of relevant dataset file according to  `~/dataset/`.

## Inspect a Model

You can refer to the *visualizeModel.ipynb* to inspect the model after training. The script use the model to predict on a grid-like dataset.

Note that you must pass the folder name where the trained model lies (the default naming is  a timestamp with a format like `yyyy-mm-dd-HHMMSS`).
