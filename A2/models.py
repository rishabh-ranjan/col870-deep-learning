import torch
from torch import nn
import torch.nn.functional as F

import utils

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
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.fc(torch.flatten(self.conv(X),-3,-1))
    
    def criterion(self, Y, y):
        return self.cross_entropy_loss(Y, y)
    
    def predict(self, X):
        return torch.argmax(self(X), dim=-1)
    
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.zero_c = ZeroClassifier()
        self.lenet = LeNet()
        self.nll_loss = nn.NLLLoss()
    
    def forward(self, X):
        Z = torch.sigmoid(self.zero_c(X))
        N = torch.softmax(self.lenet(X), dim=-1)
        return torch.cat((Z, (1-Z)*N), dim=-1)
    
    def criterion(self, Y, y):
        return self.nll_loss(torch.log(Y), y)
    
    def predict(self, X):
        return torch.argmax(self(X), dim=-1)

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
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def get_rc(self):
        t = F.one_hot(torch.arange(8))
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
    
    # (-1,9) -> (-1,8) or (-1,576) -> (-1,8)
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
    
    def pce_loss(self, Y, P):
        return torch.mean(torch.sum(-P*F.log_softmax(Y, dim=-1), dim=-1))
    
    def criterion(self, Y, P):
        # ignore Y
        self.losses = torch.empty(self.n_steps, device=P.device)
        if P.dtype == torch.long:
            for step in range(self.n_steps):
                self.losses[step] = self.cross_entropy_loss(self.out[step], P.view(-1))
        else:
            P = P[:,1:]
            P = P/torch.sum(P, dim=-1)[:,None]
            for step in range(self.n_steps):
                self.losses[step] = self.pce_loss(self.out[step], P)
        return torch.mean(self.losses)
    
    def predict(self, X):
        return torch.argmax(self(X), dim=-1).view(-1,64)+1

class DigitRRN(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.dnet = DigitNet()
        self.rrn = RRN(n_steps)
    
    def forward(self, X):
        X = utils.split_sudoku_img(X)
        return self.rrn(self.dnet(X))
    
    def dce_loss(self, Q, P):
        return torch.mean(torch.sum(-P*torch.log(torch.clamp(Q,1e-9,1e9)), dim=-1))
    
    def criterion(self, X_pred, X_true):
        # ignore X_pred
        X_true = utils.split_sudoku_img(X_true)
        P = self.dnet(X_true)
        pos_loss = self.rrn.criterion(X_pred, P)
        Q = 1-P[torch.randperm(P.shape[0])]
        neg_loss = self.dce_loss(Q, P)
        return pos_loss, neg_loss
    
    def predict(self, X):
        return torch.argmax(self(X), dim=-1).view(-1,64)+1
    
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
    
class DigitGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet = DigitNet()
        self.gen = Generator()
    
    def forward(self, X):
        return self.gen(self.dnet(X))
    
    def criterion(self, fake_yhat):
        return self.gen.criterion(fake_yhat)
    
class DigitRRNGenerator(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.dnet = DigitNet()
        self.rrn = RRN(n_steps)
        self.gen = Generator()
    
    def forward(self, X):
        X = utils.split_sudoku_img(X)
        Y = F.softmax(self.rrn(self.dnet(X)), dim=-1)
        pad = torch.zeros(*Y.shape[:-1],1, device=Y.device)
        return self.gen(torch.cat((pad, Y), dim=-1))
    
    def criterion(self, fake_yhat):
        return self.gen.criterion(fake_yhat)

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

class DigitDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dnet = DigitNet()
        self.disc = Discriminator()
        
    def forward(self, X, Y):
        return self.disc(X, self.dnet(Y))
    
    def criterion(self, real_yhat, fake_yhat):
        return self.disc.criterion(real_yhat, fake_yhat)
    
class DigitRRNDiscriminator(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.dnet = DigitNet()
        self.rrn = RRN(n_steps)
        self.disc = Discriminator()
        
    def forward(self, X, Y):
        if X.shape[-1] != 28:
            X = utils.split_sudoku_img(X)
        if Y.shape[-1] != 28:
            Y = utils.split_sudoku_img(Y)
        Y = F.softmax(self.rrn(self.dnet(Y)), dim=-1)
        pad = torch.zeros(*Y.shape[:-1],1, device=Y.device)
        return self.disc(X, torch.cat((pad, Y), dim=-1))
    
    def criterion(self, real_yhat, fake_yhat):
        return self.disc.criterion(real_yhat, fake_yhat)
    
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
