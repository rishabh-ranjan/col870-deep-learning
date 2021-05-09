import itertools as it

import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.benchmark = True
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import utils

def train_net(net, X, Y, lr, batch_size, n_epochs, device, show_step=None):
    net.train()
    net = net.to(device)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(net.parameters(), lr=lr)
    losses = []
    ctr = 0
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for X, Y in tqdm(loader, 'batches', leave=False):
            ctr += 1
            X = X.to(device)
            Y = Y.to(device)
            loss = net.criterion(net(X), Y)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            if show_step is not None and ctr % show_step == 0:
                plt.figure()
                plt.plot(losses)
                plt.xlabel('batches')
                plt.ylabel('loss')
                plt.title(f'epochs={epoch} batches={ctr} lr={lr} batch_size={batch_size}')
                plt.show()
                plt.close()
    return losses

def train_net_val(net, X, Y, val_X, val_Y, lr, batch_size, n_epochs, device, show_step=None):
    net.train()
    net = net.to(device)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(net.parameters(), lr=lr)
    losses = []
    val_losses = []
    ctr = 0
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for XY, val_XY in zip(tqdm(loader, 'batches', leave=False), it.cycle(val_loader)):
            ctr += 1
            X, Y = XY
            val_X, val_Y = val_XY
            
            with torch.no_grad():
                val_X = val_X.to(device)
                val_Y = val_Y.to(device)
                val_loss = net.criterion(net(val_X), val_Y)
                val_losses.append(val_loss.item())
                
            X = X.to(device)
            Y = Y.to(device)
            loss = net.criterion(net(X), Y)
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            
            if show_step is not None and ctr % show_step == 0:
                plt.figure()
                plt.plot(losses, label='train')
                plt.plot(val_losses, label='val')
                plt.xlabel('batches')
                plt.ylabel('loss')
                plt.title(f'epochs={epoch} batches={ctr} lr={lr} batch_size={batch_size}')
                plt.show()
                plt.close()
    return losses

def train_rrn(net, X, Y, lr, batch_size, n_epochs, device, steps=None, show_step=None):
    net.train()
    net = net.to(device)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(net.parameters(), lr=lr)
    step_losses = [[] for _ in steps]
    ctr = 0
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for X, Y in tqdm(loader, 'batches', leave=False):
            ctr += 1           
            X = X.to(device)
            Y = Y.to(device)
            loss = net.criterion(net(X), Y)
            for i, step in enumerate(steps):
                step_losses[i].append(net.losses[step].item())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            
            if show_step is not None and ctr % show_step == 0:
                plt.figure()
                for i, step in enumerate(steps):
                    plt.plot(step_losses[i], label=f'{step+1}')
                plt.legend()
                plt.xlabel('batches')
                plt.ylabel('loss')                
                plt.title(f'Step Wise Losses: epochs={epoch} lr={lr} batch_size={batch_size}')
                plt.show()
                plt.close()
    return losses

def train_rrn_val(net, X, Y, val_X, val_Y, lr, batch_size, n_epochs, device, steps, show_step=None):
    net.train()
    net = net.to(device)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(net.parameters(), lr=lr)
    losses = []
    val_losses = []
    step_losses = [[] for _ in steps]
    ctr = 0
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for XY, val_XY in zip(tqdm(loader, 'batches', leave=False), it.cycle(val_loader)):
            ctr += 1
            X, Y = XY
            val_X, val_Y = val_XY
            
            with torch.no_grad():
                val_X = val_X.to(device)
                val_Y = val_Y.to(device)
                val_loss = net.criterion(net(val_X), val_Y)
                val_losses.append(val_loss.item())
                
            X = X.to(device)
            Y = Y.to(device)
            loss = net.criterion(net(X), Y)
            losses.append(loss.item())
            for i, step in enumerate(steps):
                step_losses[i].append(net.losses[step].item())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            opt.step()
            
            if show_step is not None and ctr % show_step == 0:
                plt.figure(figsize=(12.8,4.8))
                plt.subplot(121)
                plt.plot(losses, label='train')
                plt.plot(val_losses, label='val')
                plt.legend()
                plt.xlabel('batches')
                plt.ylabel('loss')
                plt.title('Train and Val Losses')
                
                plt.subplot(122)
                for i, step in enumerate(steps):
                    plt.plot(step_losses[i], label=f'{step+1}')
                plt.legend()
                plt.xlabel('batches')
                plt.ylabel('loss')
                plt.title('Step Wise Losses')
                
                plt.suptitle(f'epochs={epoch} batches={ctr} lr={lr} batch_size={batch_size}')
                plt.show()
                plt.close()
    return losses

def train_gan_batch(X, Y, gen, disc, gen_opt, disc_opt):
    gen.train()
    disc.train()
    real_yhat = disc(X, Y)
    fake_X = gen(Y)
    fake_yhat = disc(fake_X.detach(), Y)
    
    disc_loss = disc.criterion(real_yhat, fake_yhat)
    disc_opt.zero_grad()
    disc_loss.backward()
    nn.utils.clip_grad_norm_(disc.parameters(), 1)
    disc_opt.step()
    
    fake_yhat = disc(fake_X, Y)
    gen_loss = gen.criterion(fake_yhat)
    gen_opt.zero_grad()
    gen_loss.backward()
    nn.utils.clip_grad_norm_(gen.parameters(), 1)
    gen_opt.step()
    
    return gen_loss.item(), disc_loss.item()

def train_gan(X, Y, gen, disc, lr, batch_size, n_epochs, device, show_step=None):
    gen = gen.to(device)
    disc = disc.to(device)
    gen_opt = optim.Adam(gen.parameters(), lr=lr)
    disc_opt = optim.Adam(disc.parameters(), lr=lr)
    gen_losses = []
    disc_losses = []
    ctr = 0
    loader = DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=True)
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for X, Y in tqdm(loader, 'batches', leave=False):
            ctr += 1

            X = X.to(device)
            Y = Y.to(device)
            gen_loss, disc_loss = train_gan_batch(X, Y, gen, disc, gen_opt, disc_opt) 
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

            if show_step is not None and ctr % show_step == 0:
                gen.eval()
                with torch.no_grad():
                    fake_X = gen(Y[:64,...])
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

def train_old_gan(X, Y, gen, disc, n_epochs, device, show_step=None):
    gen = gen.to(device)
    disc = disc.to(device)
    disc_opt = optim.SGD(disc.parameters(), lr=0.1, momentum=0.5)
    gen_opt = optim.SGD(gen.parameters(), lr=0.1, momentum=0.5)
    disc_lrs = optim.lr_scheduler.ExponentialLR(disc_opt, gamma=1/1.00004)
    gen_lrs = optim.lr_scheduler.ExponentialLR(gen_opt, gamma=1/1.00004)
    loader = DataLoader(TensorDataset(X, Y), batch_size=100, shuffle=True)
    gen_losses = []
    disc_losses = []
    ctr = 0
    for epoch in tqdm(range(n_epochs), 'epochs'):
        for X, Y in tqdm(loader, 'batches'):
            ctr += 1

            X = X.to(device)
            Y = Y.to(device)
            gen_loss, disc_loss = train_gan_batch(X, Y, gen, disc, gen_opt, disc_opt) 
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

            if show_step is not None and ctr % show_step == 0:
                gen.eval()
                with torch.no_grad():
                    fake_X = gen(Y[:64,...])
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
                
        gen_lrs.step()
        disc_lrs.step()
                
    return gen_losses, disc_losses
