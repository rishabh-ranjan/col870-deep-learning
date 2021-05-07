import torch
from torch import nn
import torch.nn.functional as F

import utils

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6,16,5), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256,120), nn.ReLU(),
            nn.Linear(120,84), nn.ReLU(),
            nn.Linear(84,8)
        )

    def forward(self, X):
        return self.fc(torch.flatten(self.conv(X),-3,-1))
    
    def criterion(self, Y, P):
        return utils.pce_loss(Y, P)
    
    def predict(self, X):
        return torch.argmax(self(X), dim=-1)
    
class ZeroClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(784,1)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, X):
        return self.net(X.view(-1,784))
    
    def criterion(self, Y, y):
        return self.bce_loss(Y.view(-1), y)
    
    def predict(self, X):
        return self(X)>0

class ResBlock(nn.Module):
    def __init__(self, norm_layer, in_channels, out_channels, down_sample=False):
        super().__init__()
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
    
    def forward(self, X):
        if self.in_channels == self.out_channels:
            return F.relu(X + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X))))))
        else:
            return F.relu(self.proj(X) + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X))))))

class ResNet(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
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
                layer3.append(ResBlock(self.norm_layer, 32, 64, down_sample=True))
            else:
                layer3.append(ResBlock(self.norm_layer, 64, 64))
        self.layer3 = nn.ModuleList(layer3)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 9) 
        
    # (-1,1,28,28) -> (-1,8)
    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        for layer in self.layer1:
            X = layer(X)
        for layer in self.layer2:
            X = layer(X)
        for layer in self.layer3:
            X = layer(X)
        X = self.pooling(X)
        X = X.flatten(start_dim=1) 
        return self.fc(X) 

class RRN(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps
        make_mlp = lambda i: nn.Sequential(
                                nn.Linear(i,96), nn.ReLU(),
                                nn.Linear(96,96), nn.ReLU(),
                                nn.Linear(96,96), nn.ReLU(),
                                nn.Linear(96,16)
                            )
        self.inp_enc = make_mlp(25)
        self.msg_enc = make_mlp(32)
        self.msg_comb = make_mlp(32)
        self.lstm_cell = nn.LSTMCell(16,16)
        self.decoder = nn.Linear(16,8)
        self.rc = self.get_rc()
        self.l, self.r = self.get_lr()
        
    def get_rc(self):
        t = F.one_hot(torch.arange(8, device=device))
        rc = torch.cat((t.repeat(8,1), t.repeat(1,8).view(-1,8)), dim=-1)
        return rc.float()
    
    def get_lr(self):
        s = set()
        for i in range(8):
            for j in range(8):
                start = 8*i+j
                for x in range(8):
                    end = 8*i+x
                    s.add((start,end))
                    end = 8*x+j
                    s.add((start,end))
                block_start_x = i//2*2
                block_start_y = j//4*4
                for x in range(2):
                    for y in range(4):
                        X, Y = block_start_x + x, block_start_y + y
                        end = 8*X + Y
                        s.add((start,end))
        l, r = zip(*s)
        return torch.tensor(l, dtype=torch.long), torch.tensor(r, dtype=torch.long)
    
    # (-1,9) -> (-1,8)
    def forward(self, X):
        X = X.view(-1,64,9)
        b = X.shape[0]
        RC = self.rc[None,:,:].to(X.device).expand(b,-1,-1)
        X = self.inp_enc(torch.cat((RC, X.float()), dim=-1)).view(-1,16)
        H = X
        C = torch.zeros_like(H)
        self.out = []
        for step in range(self.n_steps):
            Hv = H.view(-1,64,16)
            M = torch.zeros(b,64,64,16, device=H.device)
            M[:,self.l,self.r,:] = self.msg_enc(torch.cat((Hv[:,self.l,:], Hv[:,self.r,:]), dim=-1))
            XM = self.msg_comb(torch.cat((X, torch.sum(M, dim=-2).view(-1,16)), dim=-1))
            H, C = self.lstm_cell(XM, (H, C))
            self.out.append(self.decoder(H))
        return self.out[-1]
    
    def criterion(self, P):
        self.losses = torch.empty(self.n_steps, device=P.device)
        for step in range(self.n_steps):
            self.losses[step] = self.pce_loss(self.out[step], P)
        return torch.mean(self.losses)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        make_block = lambda i, o, k, s: nn.Sequential(
            nn.ConvTranspose2d(i, o, k, s),
            nn.BatchNorm2d(o),
            nn.ReLU()
        )
        self.gen = nn.Sequential(
            make_block(73, 256, 3, 2),
            make_block(256, 128, 4, 1),
            make_block(128, 64, 3, 2),
            nn.ConvTranspose2d(64, 1, 4, 2),
            nn.Tanh()
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    # (-1,9) -> (-1,1,28,28)
    def forward(self, Y):
        Z = torch.randn(Y.shape[0],64, device=Y.device)
        return self.gen(torch.cat((Z,Y), dim=-1).view(-1,73,1,1))
    
    def criterion(self, fake_yhat):
        return self.bce_loss(fake_yhat, torch.ones_like(fake_yhat))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        make_block = lambda i, o, k, s: nn.Sequential(
            nn.Conv2d(i, o, k, s),
            nn.BatchNorm2d(o),
            nn.LeakyReLU(0.2)
        )
        self.disc = nn.Sequential(
            make_block(10,64,4,2),
            make_block(64,128,4,2),
            nn.Conv2d(128,1,4,2)
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    # (-1,1,28,28), (-1,9) -> (-1,1)
    def forward(self, X, Y):
        return self.disc(torch.cat((X, Y[:,:,None,None].repeat(1,1,28,28)), dim=1)).view(-1,1)
    
    def criterion(self, real_yhat, fake_yhat):
        return (self.bce_loss(real_yhat, torch.ones_like(real_yhat)) +
                self.bce_loss(fake_yhat, torch.zeros_like(fake_yhat)))/2

class OldGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = nn.Sequential(nn.Linear(100,200), nn.ReLU())
        self.m2 = nn.Sequential(nn.Linear(9,1000), nn.ReLU())
        self.m3 = nn.Sequential(nn.Linear(1200,784), nn.Sigmoid())
        self.d = nn.Dropout(0.5)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    # (-1,100), (-1,9) -> (-1,1,28,28)
    def forward(self, Y):
        Z = torch.rand(Y.shape[0],100, device=Y.device)
        return self.m3(self.d(torch.cat((self.m1(Z), self.m2(Y)), dim=-1))).view(-1,1,28,28)

    def criterion(self, fake_yhat):
        return self.bce_loss(fake_yhat, torch.ones_like(fake_yhat))
    
class OldDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = nn.Sequential(nn.Linear(784,1200), nn.MaxPool1d(5,5))
        self.m2 = nn.Sequential(nn.Linear(9,250), nn.MaxPool1d(5,5))
        self.m3 = nn.Sequential(nn.Linear(290,960), nn.MaxPool1d(4,4))
        self.m4 = nn.Linear(240,1)
        self.d = nn.Dropout(0.5)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    # (-1,1,28,28), (-1,9) -> (-1,1)
    def forward(self, X, Y):
        return self.m4(self.d(self.m3(self.d(torch.cat((self.m1(X.view(-1,1,784)), self.m2(Y.view(-1,1,9))), dim=-1))))).view(-1,1)

    def criterion(self, real_yhat, fake_yhat):
        return (self.bce_loss(real_yhat, torch.ones_like(real_yhat)) +
                self.bce_loss(fake_yhat, torch.zeros_like(fake_yhat)))/2
