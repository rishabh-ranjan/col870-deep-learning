import torch
import torch.nn.functional as F
import models, train, utils
from importlib import reload
from torch.utils.data import TensorDataset, Dataset, DataLoader
import time 
import torch.optim as optim
import logging
import sys
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from PIL import Image


def generate_images(gen, device, num_images=5000):
    gen.eval()

    all_labels = torch.empty(0).long()
    all_imgs = torch.empty(0, 1, 28, 28)
    
    for label in tqdm([0,1,2,3,4,5,6,7,8], 'generating images'):
        
        a = torch.zeros(num_images).long() + label
        enc = F.one_hot(a, num_classes=9)
        images = gen(enc.to(device).float())

        all_labels = torch.cat((all_labels, a), dim=0)
        all_imgs = torch.cat((all_imgs, images.detach().cpu()), dim=0)

    return all_imgs, all_labels

logging.basicConfig(
    format='[ %(asctime)s ] %(message)s',
    level=logging.INFO
) 

reload(models)
reload(train)
reload(utils)

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True


class TruthDataset(Dataset):
    def __init__(self, l, r, path):
        
        self.X = (torch.tensor((np.load(path))[0:9])[:,None,:,:].float() - 127.5)/128.0
        self.l = self.X[l]
        self.r = self.X[r]
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1), shear=30, fillcolor=255, resample=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
        self.num_samples = 5000
        
    def __len__(self):
        return 2 * self.num_samples
    
    def __getitem__(self, i):
        if i < self.num_samples:
            return (self.augment(self.l), torch.tensor(0))
        else:
            return (self.augment(self.r), torch.tensor(1))
        
    

def generate_classifier(device, comb_X, sample_path):
    
    ensemble = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}}

    for l in range(0, 9):
        for r in range(l + 1, 9):
            
            td = TruthDataset(l, r, sample_path)
            net = models.LeNetDropout(num_classes=2)
            net = net.to(device)

            tic = time.time()
            optimizer = optim.Adam(net.parameters(), lr=4e-4)


            trainloader = torch.utils.data.DataLoader(td, batch_size=128, shuffle=True)
            running_loss = 0.0

            print("Model:", l, " ", r)
            for epoch in range(2):
                print("Starting epoch " + str(epoch) + ", time: ", time.time() - tic)
                for i, data in enumerate(trainloader, 0):
                    net.train()

                    optimizer.zero_grad()
                    images, labels = data
                    images, labels = images.to(device).float(), labels.to(device)

                    pred = net(images)
                    loss = net.criterion(pred, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 10 == 9:
                        print("epoch: ", epoch + 1, "iter: ", i + 1, "loss: ", running_loss / 10, "time: ", time.time() - tic)
                        running_loss = 0

            ensemble[l][r] = net.state_dict()
            
    actual = []
    prediction = []

    total = comb_X.shape[0]
    #total = 10000
    votes = torch.zeros(total, 9).cuda()
    
    for l in range(0, 9):
        for r in range(l + 1, 9):
            
            print("Voting on", l, r)
            net = models.LeNetDropout(num_classes=2).to(device)
            net.load_state_dict(ensemble[l][r])
            net.eval()
            with torch.no_grad():

                l_one = F.one_hot(torch.tensor(l), num_classes=9)[None,:].float().cuda()
                r_one = F.one_hot(torch.tensor(r), num_classes=9)[None,:].float().cuda()

                dataset = TensorDataset(comb_X)
                loader = torch.utils.data.DataLoader(dataset, batch_size=1024)
                masks = torch.empty(0).byte()

                i = 0
                for mbatch in loader:
                    B = mbatch[0].shape[0]
                    pred = F.softmax(net(mbatch[0].cuda()), dim=1)
                    mask = pred.max(dim=1)[0] > 0.99
                    
                    mask = mask.reshape(B, 1).float()

                    lwins = (pred[:,0] > pred[:,1]).reshape(B, 1).float()
                    rwins = (pred[:,0] < pred[:,1]).reshape(B, 1).float()

                    votes[i:i + B] += mask * (lwins * l_one + rwins * r_one) 
                    i += B
    
    mask = votes.max(dim=1)[0] > 7
    print("Elements with more than 7 votes with > 99% confidence:", mask.sum().item())
    train_X, train_y = comb_X[mask], votes[mask].argmax(dim=1)

    
    
    return train_X.detach().cpu(), train_y.detach().cpu()
    

    
if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    logging.info(f'Device: {device}')
    
    sample_path = sys.argv[1]
    
    target_X = torch.load('data/pt-cache/target_X_split.pt')
    query_X = torch.load('data/pt-cache/query_X_split.pt')    
    comb_X = torch.cat((query_X, target_X), dim=0)

    
    filter_X, filter_y = generate_classifier(device, comb_X, sys.argv[1])
    
    logging.info('Filtered dataset made')
    torch.save(filter_X.detach().cpu(), 'data/pt-cache/filter_X.pt')
    torch.save(filter_y.detach().cpu(), 'data/pt-cache/filter_y.pt')
    
    filter_X, _, filter_y, _ = utils.balanced_split(filter_X, filter_y, train_size=15000, test_size=10)

    logging.info(f'Trimming dataset to size of {filter_X.shape[0]}')

    #torch.save(filter_X, f'data/pt-cache/filter{i}_X.pt')
    #torch.save(filter_y, f'data/pt-cache/filter{i}_y.pt')
    #logging.info('Saved filtered cluster')

    filter_y = F.one_hot(filter_y, num_classes=9).float() 

    gen = models.Generator()
    disc = models.Discriminator()

    train.train_gan(filter_X, filter_y, gen, disc,
                lr=2e-4, batch_size=64, n_epochs=50,
                device=device, show_step=None)

    torch.save(gen.state_dict(), f'data/models/cgan_gen.pt')
    torch.save(disc.state_dict(), f'data/models/cgan_disc.pt')
    
    
    img, lbl = generate_images(gen, device)
    '''
    gen = models.Generator()
    disc = models.Discriminator()
    
    gen.load_state_dict(torch.load('data/models/cgan_gen.pt'))
    gen.to(device)
    
    logging.info('Training on generated images')
    '''
    
    
    net = models.LeNetDropout().to(device)

    optimizer = optim.Adam(net.parameters(), lr=3e-4,)
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(TensorDataset(img, lbl), batch_size=128, shuffle=True)
    #testloader = torch.utils.data.DataLoader(TensorDataset(test_X, test_y), batch_size=128, shuffle=False)
    
    tic = time.time()
    running_loss = 0.0
    for epoch in range(2):
        net.train()
        print("Starting epoch " + str(epoch) + ", time: ", time.time() - tic)
        for i, data in enumerate(trainloader, 0):
            net.train()
            optimizer.zero_grad()
            images, labels = data
            images, labels = images.to(device), labels.to(device)


            pred = net(images)
            loss = net.criterion(pred, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print("epoch: ", epoch + 1, "iter: ", i + 1, "loss: ", running_loss / 10, "time: ", time.time() - tic)
                running_loss = 0
    
    torch.save(net.state_dict(), 'data/models/lenet-dropout.pt')