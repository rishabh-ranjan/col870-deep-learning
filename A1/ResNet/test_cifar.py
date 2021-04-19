import argparse
import resnet
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import time
import sys
import numpy as np
import os
from normalization import NoNorm, InstanceNorm, LayerNorm, BatchNorm, BatchInstanceNorm, GroupNorm
import copy
from resnet import ResNet

parser = argparse.ArgumentParser()


class CifarDataset(data.Dataset):
    def __init__(self, path='ner-gmb/train.txt'):
        
        self.X = np.genfromtxt(path, delimiter=',')
        self.X = self.X.reshape((self.X.shape[0], 3, 32, 32))
        self.X = torch.from_numpy(self.X)
        self.X = (self.X / 255.0).float()
        self.transform = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.transform(self.X[i])
    
    

parser.add_argument('--model_file')
parser.add_argument('--normalization')
parser.add_argument('--n')
parser.add_argument('--test_data_file')
parser.add_argument('--output_file')
args = parser.parse_args()

norm_dict = {"nn": NoNorm, "bn": BatchNorm, "gn": GroupNorm,
        "in": InstanceNorm, "bin": BatchInstanceNorm, "ln": LayerNorm, "torch_bn": nn.BatchNorm2d}
    
norm = norm_dict[args.normalization]


cifar_test = CifarDataset(args.test_data_file)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=128, shuffle=False, num_workers=4)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#net = ResNet(norm_layer=norm).to(device)
net = torch.load(open(args.model_file, 'rb'), map_location=device)
net.eval()

Ypred = []
for i, X in enumerate(testloader):
    X = X.to(device)
    Y = net(X)
    Ypred.extend(Y.argmax(dim=1).tolist())

output_file = open(args.output_file, 'w')
for y in Ypred:
    output_file.write(str(y) + '\n')

output_file.close()
    