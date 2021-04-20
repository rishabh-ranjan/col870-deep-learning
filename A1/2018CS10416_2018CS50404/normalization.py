import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import math


class NoNorm(nn.Module):
    def __init__(self, num_channels=32):
        super().__init__()
        
    def forward(self, x):
        return x
    

class InstanceNorm(nn.Module):
    #InstanceNorm doesn't require running statistics like BatchNorm does
    def __init__(self, num_channels=32):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor([1e-3]))
        self.eps.requires_grad = False
        self.num_channels = num_channels
        
        self.gamma = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.ones_(self.gamma)
        
        self.beta = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.zeros_(self.beta)
        
        
    def forward(self, x):
    
        batch_mean = x.mean(dim=(2,3), keepdim=True)
        batch_var = x.var(dim=(2,3), keepdim=True)
        xnorm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        return xnorm * self.gamma + self.beta
    
    
class BatchNorm(nn.Module):
    def __init__(self, num_channels=32):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor([1e-5]))
        self.eps.requires_grad = False
        self.num_channels = num_channels
        
        self.gamma = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.ones_(self.gamma)
        
        self.beta = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.zeros_(self.beta)
        
        self.running_mean = 0
        self.running_var = 0
        self.momentum = 0.1
        
    def forward(self, x):
        
        if self.training:
            
            batch_mean = x.mean(dim=(0,2,3), keepdim=True)
            batch_var = x.var(dim=(0,2,3), keepdim=True)
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            return x_norm * self.gamma + self.beta
        else:
            
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            return x_norm * self.gamma + self.beta


        
        
class LayerNorm(nn.Module):
    #InstanceNorm doesn't require running statistics like BatchNorm does
    def __init__(self, num_channels=32):
        super().__init__()
        self.eps = torch.nn.Parameter(torch.tensor([1e-5]))
        self.eps.requires_grad = False
        self.num_channels = num_channels
        
        self.gamma = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.ones_(self.gamma)
        
        self.beta = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.zeros_(self.beta)
        
        
    def forward(self, x):
    
        batch_mean = x.mean(dim=(1,2,3), keepdim=True)
        batch_var = x.var(dim=(1,2,3), keepdim=True)
        xnorm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        return xnorm * self.gamma + self.beta
        
        
     
    

class GroupNorm(nn.Module):
    #InstanceNorm doesn't require running statistics like BatchNorm does
    def __init__(self, num_channels=32, G=8):
        super().__init__()
        
        self.G = G
        self.eps = torch.nn.Parameter(torch.tensor([1e-5]))
        self.eps.requires_grad = False
        self.num_channels = num_channels
        
        self.gamma = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.ones_(self.gamma)
        
        self.beta = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.zeros_(self.beta)
        
        
    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(N, self.G, C // self.G, H, W)
        
        group_mean = x.mean(dim=(2,3,4), keepdim=True)
        group_var = x.var(dim=(2,3,4), keepdim=True)
    
        xnorm = (x - group_mean) / torch.sqrt(group_var + self.eps)
        xnorm = xnorm.reshape(N, C, H, W)

        return xnorm * self.gamma + self.beta
    

    
    
class BatchInstanceNorm(nn.Module):
    #InstanceNorm doesn't require running statistics like BatchNorm does
    def __init__(self, num_channels=32):
        super().__init__()
        
        self.eps = torch.nn.Parameter(torch.tensor([1e-5]))
        self.eps.requires_grad = False
        self.num_channels = num_channels
        
        self.gamma = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.ones_(self.gamma)
        
        self.beta = torch.nn.Parameter(torch.empty(1,self.num_channels,1,1))
        nn.init.zeros_(self.beta)
        
        
        self.rho = torch.nn.Parameter(torch.empty(1, self.num_channels, 1, 1))
        nn.init.constant_(self.rho, 0.5)
        
        
    def forward(self, x):
        batch_mean = x.mean(dim=(0,2,3), keepdim=True)
        batch_var = x.var(dim=(0,2,3), keepdim=True)
        x_batch = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        
        instance_mean = x.mean(dim=(2,3), keepdim=True)
        instance_var = x.var(dim=(2,3), keepdim=True)
        x_instance = (x - instance_mean) / torch.sqrt(instance_var + self.eps)
        
        
        x_norm = self.rho * x_batch + (1 - self.rho) * x_instance

        return x_norm * self.gamma + self.beta