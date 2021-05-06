#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from sklearn.model_selection import train_test_split
import torch
torch.backends.cudnn.benchmark = True
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms, utils
from tqdm.auto import tqdm
import time


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[3]:


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
    
    def forward(self, x):
        if self.in_channels == self.out_channels:
            return F.relu(x + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        else:
            return F.relu(self.proj(x) + self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

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
        return self.fc(x)


# In[4]:


class RRN(nn.Module):
    def __init__(self, n_steps, step_loss=False):
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
        self.step_loss = step_loss
        
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
    
    def forward(self, X):
        b = X.shape[0]
        X = X.view(-1,64,9)
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
            O = self.decoder(H)
            if self.step_loss:
                self.out.append(O)
        return O.view(-1,512)
    
    def criterion(self, y_true, X_pred):
        if self.step_loss:
            self.losses = torch.empty(self.n_steps, device=y_true.device)
            for step in range(self.n_steps):
                self.losses[step] = self.cross_entropy_loss(self.out[step], y_true.view(-1))
            return torch.sum(self.losses)
        else:
            return self.cross_entropy_loss(X_pred.view(-1,8), y_true.view(-1))


# In[5]:


class Generator(nn.Module):
    def __init__(self, z_dim, n_classes):
        super().__init__()
        make_block = lambda i, o, k, s: nn.Sequential(
            nn.ConvTranspose2d(i, o, k, s),
            nn.BatchNorm2d(o),
            nn.ReLU()
        )
        self.gen = nn.Sequential(
            make_block(z_dim+n_classes, 256, 3, 2),
            make_block(256, 128, 4, 1),
            make_block(128, 64, 3, 2),
            nn.ConvTranspose2d(64, 1, 4, 2),
            nn.Tanh()
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.z_dim = z_dim
        self.n_classes = n_classes

    def forward(self, Z, Y):
        return self.gen(torch.cat((Z,Y), dim=-1).view(-1,self.z_dim+self.n_classes,1,1))
    
    def criterion(self, fake_yhat):
        return self.bce_loss(fake_yhat, torch.ones_like(fake_yhat))


# In[6]:


class JointGenerator(nn.Module):
    def __init__(self, resnet, n_steps, z_dim, w):
        super().__init__()
        self.resnet = resnet
        self.rrn = RRN(n_steps, step_loss=False)
        self.gen = Generator(z_dim, 8)
        self.z_dim = z_dim
        self.l1_loss = nn.L1Loss()
        self.w = w
        
    def forward(self, X):
        Y = self.rrn(self.resnet(X).view(-1,576)).view(-1,8)
        Z = torch.randn(Y.shape[0], self.z_dim, device=Y.device)
        return self.gen(Z, Y)
    
    def criterion(self, fake_yhat, real_X, fake_X):
        return self.gen.criterion(fake_yhat) +  self.w * self.l1_loss(real_X, fake_X)


# In[7]:


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        make_block = lambda i, o, k, s: nn.Sequential(
            nn.Conv2d(i, o, k, s),
            nn.BatchNorm2d(o),
            nn.LeakyReLU(0.2)
        )
        self.disc = nn.Sequential(
            make_block(1+n_classes,64,4,2),
            make_block(64,128,4,2),
            nn.Conv2d(128,1,4,2)
        )
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, X, Y):
        return self.disc(torch.cat((X, Y[:,:,None,None].repeat(1,1,28,28)), dim=1)).view(-1,1)
    
    def criterion(self, real_yhat, fake_yhat):
        return (self.bce_loss(real_yhat, torch.ones_like(real_yhat)) +
                self.bce_loss(fake_yhat, torch.zeros_like(fake_yhat)))/2


# In[8]:


def load_sudoku_images(path, total, device, normalize=False):
    sudoku_img = torch.empty(total,1,224,224, device=device)
    if normalize:
        transform = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))))
    else:
        transform = transforms.ToTensor()
    for i in tqdm(range(total), 'sudoku images'):
        sudoku_img[i,0] = transform(Image.open(f'{path}/{i}.png'))
    return sudoku_img


# In[9]:


# query_X = load_sudoku_images('data/query', 10000, device, normalize=True)
# torch.save(query_X, 'data/pt-cache/query_X.pt')
query_X = torch.load('data/pt-cache/query_X.pt')


# In[10]:


# target_X = load_sudoku_images('data/target', 10000, device, normalize=True)
# torch.save(target_X, 'data/pt-cache/target_X.pt')
target_X = torch.load('data/pt-cache/target_X.pt')


# In[11]:


def split_sudoku_img(sudoku_img):
    return torch.stack(torch.split(
        torch.stack(torch.split(sudoku_img, [28]*8, dim=-2), dim=-3),
        [28]*8, dim=-1), dim=-3).view(-1,1,28,28)


# In[12]:


def arrange_sudoku(img):
    return utils.make_grid(img, nrow=8, padding=0).view(-1,1,224,224)


# In[13]:


def viz_images(img, nrow):
    plt.imshow(utils.make_grid(((img+1)/2).detach().cpu(), nrow=nrow, padding=0).permute(1,2,0))


# In[16]:


resnet = ResNet().to(device)
resnet.load_state_dict(torch.load('data/pt-cache/resnet.pt'))
for p in resnet.parameters():
    p.requires_grad=False
gen = JointGenerator(resnet=resnet, n_steps=16, z_dim=64, w=1).to(device)
disc = Discriminator(n_classes=8).to(device)
gen_opt = optim.Adam(gen.parameters())
disc_opt = optim.Adam(disc.parameters())


# In[17]:


ctr = 0
gen_losses = []
disc_losses = []


# In[ ]:


loader = DataLoader(TensorDataset(query_X, target_X), batch_size=32, shuffle=True)
gen_opt = optim.Adam(gen.parameters(), lr=1e-4)
disc_opt = optim.Adam(disc.parameters(), lr=1e-4)
show_times = False

while True:
    for X, real_X in tqdm(loader, 'batches'):
        ctr += 1
        
        tic = time.time()
        X = split_sudoku_img(X.to(device))
        real_X = split_sudoku_img(real_X.to(device))
        toc = time.time()
        if show_times:
            print('move+split:\t', toc-tic)
        
        tic = time.time()
        fake_X = gen(X)
        toc = time.time()
        if show_times:
            print('gen:\t\t', toc-tic)
        
        tic = time.time()
        y = resnet(real_X)[...,1:]
        toc = time.time()
        if show_times:
            print('resnet:\t\t', toc-tic)
        
        tic = time.time()
        real_yhat = disc(real_X, y)
        fake_yhat = disc(fake_X.detach(), y)
        toc = time.time()
        if show_times:
            print('disc:\t\t', toc-tic)
        
        tic = time.time()
        disc_loss = disc.criterion(real_yhat, fake_yhat)
        disc_losses.append(disc_loss.item())
        disc_opt.zero_grad()
        disc_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(disc.parameters(), 1)
        disc_opt.step()
        toc = time.time()
        if show_times:
            print('disc back:\t', toc-tic)
        
        tic = time.time()
        fake_yhat = disc(fake_X, y)
        toc = time.time()
        if show_times:
            print('disc:\t\t', toc-tic)
        
        tic = time.time()
        gen_loss = gen.criterion(fake_yhat, real_X, fake_X)
        gen_losses.append(gen_loss.item())
        gen_opt.zero_grad()
        gen_loss.backward()
        nn.utils.clip_grad_norm_(gen.parameters(), 1)
        gen_opt.step()
        toc = time.time()
        if show_times:
            print('gen back:\t', toc-tic)
            print('---')
#             print('gen:',gen_losses[-1],'\t','disc:',disc_losses[-1])

        tqdm.write(f'ctr: {ctr}\tgen: {gen_losses[-1]:.3f}\tdisc: {disc_losses[-1]:.3f}')
        if ctr % 50 == 0:
            plt.figure(figsize=(15,5))
            plt.subplot(131)
            viz_images(X[:64,:,:,:], nrow=8)
            plt.subplot(132)
            viz_images(real_X[:64,:,:,:], nrow=8)
            plt.subplot(133)
            viz_images(fake_X[:64,:,:,:], nrow=8)
            plt.savefig(f'runlogs/sample.{ctr}.png')
            plt.close()
            
            plt.figure()
            plt.plot(gen_losses, label='gen')
            plt.plot(disc_losses, label='disc')
            plt.legend()
            plt.xlabel('batches')
            plt.ylabel('loss')
            plt.title(f'Loss Curve (batch_size={loader.batch_size})')
            plt.savefig(f'runlogs/loss.{ctr}.png')
            plt.close()

    torch.save(gen.state_dict(), f'data/pt-cache/all_gen.{ctr}.pt')
    torch.save(disc.state_dict(), f'data/pt-cache/all_disc.{ctr}.pt')


# In[ ]:




