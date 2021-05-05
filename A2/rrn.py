import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import time
import sys
import matplotlib.image as mpimg
import os
import copy
import torch.utils.data as torch_data
import matplotlib.pyplot as plt
import sys

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True

class RRN(nn.Module):
    
    def __init__(self):
        super(RRN, self).__init__()
        
        self.input_encoder = nn.Sequential(
                                nn.Linear(25, 96), nn.ReLU(),
                                nn.Linear(96, 96), nn.ReLU(),
                                nn.Linear(96, 96), nn.ReLU(),
                                nn.Linear(96, 16)
                            )
        
        self.msg_encoder = nn.Sequential(
                                nn.Linear(32, 96), nn.ReLU(),
                                nn.Linear(96, 96), nn.ReLU(),
                                nn.Linear(96, 96), nn.ReLU(),
                                nn.Linear(96, 16)
                            )
        
        self.msg_combiner = nn.Sequential(
                                nn.Linear(32, 96), nn.ReLU(),
                                nn.Linear(96, 96), nn.ReLU(),
                                nn.Linear(96, 96), nn.ReLU(),
                                nn.Linear(96, 16)
                            )
        
        self.lstm_cell = nn.LSTMCell(16, 16)
        self.decoder = nn.Linear(16, 8)
        
        self.adj_mask = self.generate_mask()
    
    def generate_mask(self):
        mask = torch.zeros(64, 64)
        for i in range(8):
            for j in range(8):
                start = 8 * i + j
                for x in range(8):
                    end = 8 * i + x
                    mask[start][end] = 1
                    end = 8 * x + j
                    mask[start][end] = 1
                
                block_start_x = i // 2 * 2
                block_start_y = j // 4 * 4
                
                for x in range(2):
                    for y in range(4):
                        X, Y = block_start_x + x, block_start_y + y
                        end = 8 * X + Y
                        mask[start][end] = 1
        return mask > 0
        
    
    def forward(self, h_prev, s_prev, x, m):
        '''
            h_prev (B, 8, 8, 16)
            s_prev (B, 8, 8, 16)
            x (B, 8, 8, 16)
            m (B, 8, 8, 16)
        '''
        B = h_prev.shape[0]
        xm = self.msg_combiner(torch.cat((x, m), dim=3))
        h_next, s_next = self.lstm_cell(xm.flatten(0, 2), (h_prev.flatten(0, 2), s_prev.flatten(0, 2)))
        
        o = self.decoder(h_next.reshape(-1, h_next.shape[-1])) # (B * 64 * 64, 16)
        o = o.reshape(B, 8, 8, 8) # (B, 64, 64, 16)
        h_next, s_next = h_next.reshape(B, 8, 8, 16), s_next.reshape(B, 8, 8, 16)
        
        return o, h_next, s_next
    
    def message_passing(self, h):
        B = h.shape[0]
        # h (B, 8, 8, 16)
        h_ = h.flatten(1, 2) 
        #h_ (B, 4096, 16)
        
        M_all = torch.cat((h_[:,:,None,:].repeat(1, 1, 64, 1), h_[:,None,:,:].repeat(1, 64, 1, 1)), dim=3)
        M_all = M_all.flatten(1, 2)
        # M_all (B, 4096, 32)
        all_pairs = M_all[:,self.adj_mask.flatten(),:]
        # (B, #constraints, 32)
        
        msg_pairs = self.msg_encoder(all_pairs.flatten(0, 1)).reshape(B, all_pairs.shape[1], 16)
        # (B, #constraints, 16)
        
        all_msgs = torch.zeros(B, 4096, 16).cuda()
        all_msgs[:,self.adj_mask.flatten(),:] = msg_pairs
        
        all_msgs = all_msgs.reshape(B, 64, 64, 16)
        
        return all_msgs.sum(dim=2).reshape(B, 8, 8, 16)
    
    def encode_input(self, sudoku):
        # sudoku (B, 8, 8)
        B = sudoku.shape[0]
        col = torch.arange(0, 8, 1)
        col = col[None,:].repeat(8, 1)[None,:,:].repeat(B, 1, 1)
        row = col.transpose(1, 2)
        
        row_one_hot = F.one_hot(row, num_classes=8).cuda()
        col_one_hot = F.one_hot(col, num_classes=8).cuda()
        val_one_hot = F.one_hot(sudoku, num_classes=9).cuda()
        
        input_ = torch.cat((row_one_hot, col_one_hot, val_one_hot), dim=3)
        return self.input_encoder(input_.float())
    
        
        
        

class SudokuDataset(torch_data.Dataset):
    def __init__(self):
        self.query_pred = torch.tensor(torch.load('Assignment 2/visual_sudoku/query_pred.pt'))
        self.target_pred = torch.tensor(torch.load('Assignment 2/visual_sudoku/target_pred.pt'))
        
    def __len__(self):
        return self.query_pred.shape[0] // 64
    
    def __getitem__(self, i):
        query = self.query_pred[64*i : 64*(i + 1)].reshape(8, 8)
        target = self.target_pred[64*i : 64*(i + 1)].reshape(8, 8)
        return query, target



dataset = SudokuDataset()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
print(device)
net = RRN().to(device)
print(net)

n_steps = int(sys.argv[1])
num_epochs = int(sys.argv[2])
expt_desc = "step_{}_correct.pth".format(n_steps)

print("Steps:", n_steps)

tic = time.time()


lmbda = lambda epoch: 0.90
optimizer = optim.Adam(net.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lmbda)

criterion = nn.CrossEntropyLoss()

split = int(0.8 * len(dataset))
train_data, val_data = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=256, shuffle=False)



running_loss = 0.0
for epoch in range(num_epochs):
    net.train()
    print("Starting epoch " + str(epoch) + ", time: ", time.time() - tic)
    for i, data in enumerate(train_loader, 0):
        
        
        if i == 0:
            net.eval()
            with torch.no_grad():
                total, correct = 0, 0
                total_z, correct_z = 0, 0
                for j, data in enumerate(val_loader):
                    
                    query, target = data
                    query, target = query.to(device), target.to(device) - 1
                    B = query.shape[0]
                    
                    x = net.encode_input(query)
                    s = torch.zeros(B, 8, 8, 16).to(device)

                    h = x
                    for steps in range(n_steps):
                        m = net.message_passing(h)
                        o, h, s = net(h, s, x, m)
                    del h, s
                    
                    pred = o.argmax(dim=3)
                    
                    total += query.nelement()
                    correct += (pred == target).sum()
                    
                    total_z += (query == 0).sum()
                    correct_z += ((pred == target) * (query == 0)).sum()
                    
                    #print(query.flatten(), "\n", target.flatten(), "\n", pred.flatten())
                    
                    #print(query.nelement(), (pred == target).sum().item(), (query == 0).sum().item(), ((pred == target) * (query == 0)).sum().item())
                    
                print("%(z):", (correct_z.item() / total_z.item()) * 100.0,
                     "%(t):", (correct.item() / total) * 100.0)
        
        
        net.train()
        optimizer.zero_grad()
        query, target = data
        query, target = query.to(device), target.to(device) - 1
        B = query.shape[0]
        
        x = net.encode_input(query)
        s = torch.zeros(B, 8, 8, 16).to(device)
        h = x
        loss = 0.0
        for j in range(n_steps):
            m = net.message_passing(h)
            o, h, s = net(h, s, x, m)
            loss += criterion(o.flatten(0, 2), target.flatten(0, 2))
        
        del o, h, s
        torch.cuda.empty_cache()
        loss = loss / n_steps
        
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        
        optimizer.step()
        running_loss += loss.item()
        
        
        if i % 10 == 9:
            print("epoch: ", epoch + 1, "iter: ", i + 1, "loss: ", running_loss / 10, "time: ", time.time() - tic, flush=True)
            running_loss = 0
            
        
                
                
    scheduler.step()       
    running_loss = 0.0
    
torch.save(net.state_dict(), expt_desc)
