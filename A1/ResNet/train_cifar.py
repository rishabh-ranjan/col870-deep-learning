import argparse
import resnet
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import time
import sys
import os
from normalization import NoNorm, InstanceNorm, LayerNorm, BatchNorm, BatchInstanceNorm, GroupNorm
import copy

parser = argparse.ArgumentParser()



parser.add_argument('--normalization')
parser.add_argument('--data_dir')
parser.add_argument('--output_file')
parser.add_argument('--n')
args = parser.parse_args()

resnet.train(n=args.n, norm_type=args.normalization, data_dir=args.data_dir, output_file=args.output_file, verbose=False)
