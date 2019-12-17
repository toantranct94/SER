from __future__ import division
from sklearn.model_selection import train_test_split
# import dataloader

import os
import sys

sys.path.append(os.getcwd())

from data_aug import MyDatasetSTFT
import torch
import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os, sys
from config import *
from time import time, strftime

import predicts
from utils import *
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices=[18, 50, 152], type=int, help='depth of model')
parser.add_argument('--model_path', type=str, default=' ')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_tests', type=int, default=10, help='number of tested windows in each file')
args = parser.parse_args()

path = './saved_model/'
# model preparation 

if __name__ == "__main__":
    model_path_fns = [path + 'resnet_18_1128_1429_r225.t7', path + 'resnet_18_1128_1443_r225.t7', path + 'resnet_18_1128_1448_r225.t7']
    # load models
    models = []
    clippeds = []
    for model_path_fn in model_path_fns:
        checkpoint = torch.load(model_path_fn, map_location=lambda storage, loc: storage)
        model = checkpoint['model']
        model = unparallelize_model(model)
        model = parallelize_model(model)
        best_acc = checkpoint['acc']
        clipped = checkpoint['args'].duration
        print('model {}, accuracy: {}'.format(model_path_fn, best_acc))
        models.append(model)
        clippeds.append(clipped)

    # build dset_loaders  
    print('Data preparation')
    fns = [os.path.join(BASE_PUBLIC_TEST, fn) for fn in os.listdir(BASE_PUBLIC_TEST)]
    print('Total provided files: {}'.format(len(fns)))
    lbs = [-1] * len(fns)  # random labels, we won't use this
    dset_loaders = []
    for clipped in clippeds:
        dset = MyDatasetSTFT(fns, lbs, duration=clipped, test=True)
        dset_loaders.append(torch.utils.data.DataLoader(dset,
                                                        batch_size=args.batch_size,
                                                        shuffle=False, num_workers=NUM_WORKERS))

    # make predictions based on models and loaders
    pred_score, pred_probs, fns = predicts.multimodels_multiloaders_class(models, dset_loaders, num_tests=args.num_tests)

    # write to csv file
    fns = [fn.split('/')[-1][:-4] + '.wav' for fn in fns]
    file_name ='./result/ser.csv'
    data = list(zip(fns, pred_score))
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=None, header=False, sep = ',')
    print('done')
