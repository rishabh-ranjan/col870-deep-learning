import models, train
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import utils
import logging
import torch.nn.functional as F
import sys

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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

query_X = torch.load('data/pt-cache/query_X.pt', map_location='cpu')
target_X = torch.load('data/pt-cache/target_X.pt', map_location='cpu')

samples = (torch.tensor(np.load(sys.argv[1])).float()/127.5-1)[:9,None,:,:]

net = models.LeNetRRN(24, 0.5, 0.5)
net.lenet.load_state_dict(torch.load('data/models/lenet-dropout.pt', map_location='cpu'))
net.rrn.load_state_dict(torch.load('data/models/rrn.pt', map_location='cpu'))

train_X, test_X = torch.split(query_X, [10000,0], dim=0)
train_Y, test_Y = torch.split(target_X, [10000,0], dim=0)

train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=32, shuffle=True)
aug_loader = DataLoader(models.AugmentedSamples(samples), batch_size=64)

train.train_lenet_rrn(net, train_loader, aug_loader, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0),
                      lr=1e-3, n_epochs=50, device=device, show_step=10)

torch.save(net.state_dict(), 'data/models/lenet_rrn.pt')
torch.save(net.lenet.state_dict(), 'data/models/lenet_adv.pt')
torch.save(net.rrn.state_dict(), 'data/models/rrn_adv.pt')


net = models.LeNetRRN(24, 0.5, 0.5)
net.load_state_dict(torch.load('data/models/lenet_adv.pt'))

#Generating GAN
query_X_split = torch.load('data/pt-cache/query_X_split.pt')
filter_X, filter_y = query_X_split, utils.decode_sudoku_img(query_X, net.lenet, 'cuda', 128).flatten()

filter_X, _, filter_y, _ = utils.balanced_split(filter_X, filter_y, train_size=15000, test_size=10)

logging.info(f'Trimming dataset to size of {filter_X.shape[0]}')


filter_y = F.one_hot(filter_y.long(), num_classes=9).float() 

gen = models.Generator()
disc = models.Discriminator()

train.train_gan(filter_X, filter_y, gen, disc,
            lr=2e-4, batch_size=64, n_epochs=50,
            device=device, show_step=None)

torch.save(gen.state_dict(), f'data/models/cgan_gen_adv.pt')
torch.save(disc.state_dict(), f'data/models/cgan_disc_adv.pt')
