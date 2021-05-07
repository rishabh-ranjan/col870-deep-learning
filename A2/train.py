import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.benchmark = True
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import utils

def train_gan_batch(X, Y, gen, disc, gen_opt, disc_opt):
    gen.train()
    disc.train()
    real_yhat = disc(X, Y)
    fake_X = gen(Y)
    fake_yhat = disc(fake_X.detach(), Y)
    
    disc_loss = disc.criterion(real_yhat, fake_yhat)
    disc_opt.zero_grad()
    disc_loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm_(disc.parameters(), 1)
    disc_opt.step()
    
    fake_yhat = disc(fake_X, Y)
    gen_loss = gen.criterion(fake_yhat)
    gen_opt.zero_grad()
    gen_loss.backward()
    nn.utils.clip_grad_norm_(gen.parameters(), 1)
    gen_opt.step()
    
    return gen_loss.item(), disc_loss.item()

def train_gan(X, y, gen, disc, lr, batch_size, n_epochs, device, show_step=None):
    gen = gen.to(device)
    disc = disc.to(device)
    gen_opt = optim.Adam(gen.parameters(), lr=lr)
    disc_opt = optim.Adam(disc.parameters(), lr=lr)
    gen_losses = []
    disc_losses = []
    ctr = 0
    loader = DataLoader(TensorDataset(X,y), batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for X, y in tqdm(loader, 'batches'):
            ctr += 1

            X = X.to(device)
            Y = F.one_hot(y.to(device), num_classes=9).float()
            gen_loss, disc_loss = train_gan_batch(X, Y, gen, disc, gen_opt, disc_opt) 
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

            if show_step is not None and ctr % show_step == show_step-1:
                gen.eval()
                with torch.no_grad():
                    fake_X = gen(Y[:64,:])
                plt.figure(figsize=(12,5))
                plt.subplot(121)
                utils.viz_images(X[:64], nrow=8)
                plt.subplot(122)
                utils.viz_images(fake_X, nrow=8)
                plt.show()
                plt.close()

                plt.figure()
                plt.plot(disc_losses, label='disc')
                plt.plot(gen_losses, label='gen')
                plt.legend()
                plt.xlabel('batches')
                plt.ylabel('loss')
                plt.title('Loss Curves')
                plt.show()
                plt.close()

    return gen_losses, disc_losses