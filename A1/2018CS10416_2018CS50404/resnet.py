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

class ResBlock(nn.Module):
    def __init__(self, norm_layer, in_channels, out_channels, down_sample=False):
        
        super(ResBlock, self).__init__()
        self.norm_layer = norm_layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if down_sample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = norm_layer(out_channels)
        self.bn2 = norm_layer(out_channels)
        
        if not self.in_channels == self.out_channels:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2)
    
    def forward(self, x):
        if self.in_channels == self.out_channels:
            return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        else:
            return F.relu(self.proj(x) + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

class ResNet(nn.Module):
    def __init__(self, n=2, r=10, norm_layer=None):
        
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = self.norm_layer(16)
        
        layer1 = []
        for i in range(n):
            layer1.append(ResBlock(self.norm_layer, 16, 16))
        
        self.layer1 = nn.ModuleList(layer1)
        
        layer2 = []
        for i in range(n):
            if i == 0:
                layer2.append(ResBlock(self.norm_layer, 16, 32, down_sample=True))
            else:
                layer2.append(ResBlock(self.norm_layer, 32, 32))
        self.layer2 = nn.ModuleList(layer2)
        
        layer3 = []
        for i in range(n):
            if i == 0:
                layer3.append(ResBlock(norm_layer, 32, 64, down_sample=True))
            else:
                layer3.append(ResBlock(norm_layer, 64, 64))
        self.layer3 = nn.ModuleList(layer3)
        
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, r)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        for layer in self.layer1:
            x = layer(x)
        
        for layer in self.layer2:
            x = layer(x)
        
        for layer in self.layer3:
            x = layer(x)
            
        x = self.pooling(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        return F.softmax(x, dim=1)

    

            
            

def train(n, norm_type, data_dir, output_file, verbose=False):
    norm_dict = {"nn": NoNorm, "bn": BatchNorm, "gn": GroupNorm,
        "in": InstanceNorm, "bin": BatchInstanceNorm, "ln": LayerNorm, "torch_bn": nn.BatchNorm2d}
    
    norm = norm_dict[norm_type]


    transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    dataset = torchvision.datasets.CIFAR10(root=os.path.join(data_dir, '..'), train=True, transform=transform)
        
    trainset, valset = torch.utils.data.random_split(
        dataset, [40000, 10000])
    

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=128, shuffle=False, num_workers=4)
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    net = ResNet(norm_layer=norm).to(device)
    print(net, flush=True)
    criterion = nn.CrossEntropyLoss()

    print("-----------------------------STARTING TRAINING----------------------------------")

    tic = time.time()

    iteration = 0
    
    #optimizer = optim.Adam(net.parameters(), lr=3e-3)
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    
    validation_loss = []
    lowest_val_loss = float('inf')
    patience = 30
    steps_left = patience
    best_model = None
    
    for epoch in range(1, 201):  # loop over the dataset multiple times
        with torch.no_grad():
            net.eval()
            
            actual, prediction = [], []
            total_loss=0
            for i, data in enumerate(valloader, 0):
                inputs, labels=data
                inputs, labels=inputs.to(device), labels.to(device)

                outputs=net(inputs)
                actual.extend(labels.squeeze().tolist())
                val_loss=criterion(outputs, labels)

                prediction.extend(torch.argmax(
                    outputs, dim=1).squeeze().tolist())
                total_loss += inputs.shape[0] * val_loss.item()

            print("iteration:", iteration, " val loss:", total_loss / 10000, " val accuracy:", accuracy_score(actual, prediction), " val f1(macro)", f1_score(actual, prediction, average='macro'), " val f1(micro)", f1_score(actual, prediction, average='micro'), flush=True)
            
            curr_loss = total_loss / 10000
            validation_loss.append(curr_loss)
            print(curr_loss, steps_left, lowest_val_loss)
            if curr_loss < lowest_val_loss:
                lowest_val_loss = curr_loss
                best_model = copy.deepcopy(net.state_dict())
                steps_left = patience
            else:
                if steps_left == 1:
                    print("Patience ended")
                    break
                else:
                    steps_left -= 1
            
            

        print("Epoch: ", epoch, " time:", time.time() - tic, flush=True)

        net.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iteration += 1

            if i % 50 == 49:
                print('[%d, %5d] loss: %f iteration:%d' %
                      (epoch, i + 1, running_loss / 50, iteration), flush=True)
                running_loss = 0.0
            
        scheduler.step()

    if not best_model == None:
        net = ResNet(norm_layer=norm).to(device)
        net.load_state_dict(best_model)
    
    print("------------------------------FINAL-------------------------------")
    

    torch.save(net, output_file)
