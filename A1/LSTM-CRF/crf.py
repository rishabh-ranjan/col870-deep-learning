import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
import torch.nn.utils
import torch.optim as optim
import torch.utils.data as data
from tqdm.auto import tqdm
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import itertools as it
import time
import numpy as np
from seqeval.metrics import classification_report


def load_emb(path, total=None):
    toks = []
    embs = []
    with open(path, 'r') as f:
        for l in tqdm(f, path, total=total):
            tok, *emb = l.strip().split()
            emb = [float(x) for x in emb]
            toks.append(tok)
            embs.append(emb)
    assert('PAD_TOK' not in toks and 'UNK_TOK' not in toks)
    toks += ['PAD_TOK', 'UNK_TOK']
    embs += [[0.]*len(emb), [0.]*len(emb)]
    tok_to_id = dict(zip(toks, it.count()))
    emb = torch.tensor(embs)
    return tok_to_id, emb

# load characters from (training) data
def load_chrs(path, total=None):
    chars = set()
    with open(path, 'r') as f:
        for l in tqdm(f, path, total=total):
            try:
                for c in l.strip().split()[2]:
                    chars.add(c)
            except:
                pass
    assert('PAD_CHR' not in chars and 'UNK_CHR' not in chars)
    chars = sorted(chars)
    chars.append('PAD_CHR')
    chars.append('UNK_CHR')
    return dict(zip(chars, it.count()))          

def load_classes(path, total=None):
    id_to_lbl = set()
    with open(path, 'r') as f:
        for l in tqdm(f, path, total=total):
            try:
                id_to_lbl.add(l.strip().split()[3])
            except:
                pass
    assert('PAD_LBL' not in id_to_lbl)
    id_to_lbl = sorted(id_to_lbl)
    id_to_lbl.append('PAD_LBL')
    lbl_to_id = {k:v for v, k in enumerate(id_to_lbl)}
    return lbl_to_id, id_to_lbl
    
def load_data(path, tok_to_id, lbl_to_id, chr_to_id):
    with open(path, 'r') as f:
        seqs = f.read().split('\n\n')
        if not seqs[-1].strip():
            seqs.pop()
        if seqs[0][0] == '\n':
            seqs[0] = seqs[0][1:]
    seqs = [l.split('\n') for l in seqs]
    seq_len = max((len(seq) for seq in seqs))
    seqs = [[l.split(' ') for l in seq] for seq in seqs]
    wrd_len = max((max((len(cols[2]) for cols in seq)) for seq in seqs))
    W = torch.empty((len(seqs), seq_len, wrd_len), dtype=torch.long).fill_(chr_to_id['PAD_CHR'])
    X = torch.empty((len(seqs), seq_len), dtype=torch.long).fill_(tok_to_id['PAD_TOK'])
    Y = torch.empty((len(seqs), seq_len), dtype=torch.long).fill_(lbl_to_id['PAD_LBL'])
    for i, seq in enumerate(tqdm(seqs, 'sequences')):
        for j, cols in enumerate(seq):
            assert(j < seq_len)
            tok, _, wrd, lbl = cols
            for k, ch in enumerate(wrd):
                try:
                    W[i,j,k] = chr_to_id[ch]
                except KeyError:
                    W[i,j,k] = chr_to_id['UNK_CHR']
            try:
                X[i,j] = tok_to_id[tok]
            except KeyError:
                X[i,j] = tok_to_id['UNK_TOK']   
            Y[i,j] = lbl_to_id[lbl]
    return W, X, Y

class NERDataset(data.Dataset):
    def __init__(self, W, X, Y):
        self.W, self.X, self.Y = W, X, Y
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, i):
        return self.W[i], self.X[i], self.Y[i]

class LinearCRF(nn.Module):
    def __init__(self, input_size, hidden_size, lbl_to_id, lstm_model):
        super().__init__()
        
        #self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.lstm = lstm_model
        self.lbl_to_id = lbl_to_id
        self.num_tags = 19 #Includes the 17 tags + start + end(PAD_TOK)
        self.T = nn.Parameter(torch.empty((19, 19)))
        #self.T.data[0:17,0:17] = torch.from_numpy(get_matrix().transpose())
        nn.init.normal_(self.T)
        #Make changes here. 
        
        #17 tag is start and 18 is stop. Arriving on start, beginning on stop is very improbable
        #self.T.data[lbl_to_id['START_LBL'],:] = -1000 
        #self.T.data[:,lbl_to_id['PAD_LBL']] = -1000
        
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Sequential(  nn.Linear(2 * input_size, input_size), 
                                    nn.ReLU(),
                                    nn.Linear(input_size, 19))
        
        
    def forward(self, W, X):
        #X is of the shape (batch_size, seq_length, num_features)
        
        o = self.lstm(W, X)
        
        #return F.softmax(self.proj(o), dim=2)
        return o

    def predict(self, P, mask):
        
        with torch.no_grad():
            batch_size = P.shape[0]
            path = []
            for i in range(batch_size):
                path.append([])
                
            choice = torch.zeros(P.shape[0], P.shape[1], 19).cuda()
            
            prob_matrix = self.T
            # X.shape is (batch_size, sentence_length)
            
                    
            DP = torch.full((P.shape[0], 19), -1000).cuda()
            DP[:, self.lbl_to_id['START_LBL']] = 0 #start tag
            for i in range(batch_size):
                for j in range(19):
                    choice[i,0,j] = j
            #path[:,0] = DP[:,0,:].argmax(dim=1)
            
            for i in range(0, P.shape[1]):
                #next_DP = 
                submask = mask[:, i].unsqueeze(1).float()  # [B, 1]
                emission_score = P[:, i]  # [B, C]

                # [B, 1, C] + [C, C]
                next_choice = DP.unsqueeze(1) + self.T  # [B, C, C]
                next_choice, choice[:, i, :] = next_choice.max(dim=-1)
                next_choice += emission_score
                DP = next_choice * submask + DP * (1 - submask)  # max_score or acc_score_t
                
            DP += self.T[self.lbl_to_id['PAD_LBL']]
            last_elem = DP.argmax(-1)
            # now, the choice vector has been constructed and the solution can
            # be computed in the reverse direction. 
            # DP[i][j][k] indicates the choice made at the kth step in the jth token 
            # of the ith sentences
            
            choice = choice.cpu()
            
            for i in range(batch_size):
                
                num_tags = mask[i].sum()
                path[i].append(last_elem[i].int().item())
                prev = last_elem[i].int().item()
                for j in range(int(num_tags) - 2, -1, -1):
                    
                    path[i].append(choice[i][j + 1][prev].int().item())
                    prev = choice[i][j + 1][prev].int().item()
                    
            
            for i in range(batch_size):
                path[i].reverse()
                
        return path
    
def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()
   
def train(train_set, dev_set, ner_model, id_to_lbl, lbl_to_id, pad_lbl_id, output_file):
    
    id_to_lbl[len(id_to_lbl) - 1] = 'START_LBL'
    
    trainset = NERDataset(train_set[0], train_set[1], train_set[2])
    devset = NERDataset(dev_set[0], dev_set[1], dev_set[2])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    devloader = torch.utils.data.DataLoader(devset, batch_size=128, shuffle=False, num_workers=4)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    net = LinearCRF(100, 100, lbl_to_id, ner_model).to(device)
    
    print(net)
    
    tic = time.time()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    
    patience = 15
    steps_left = patience
    min_val_loss = float('inf')
    best_model_dict = None
    early_stop = False

    for epoch in range(1, 101):  # loop over the dataset multiple times
        #if epoch % 10 == 0 or epoch == 1:
        net.train()
        
        running_loss = 0.0
        for i, mbatch in enumerate(trainloader):
            W, X, Y = mbatch
            W, X, Y = W.to(device), X.to(device), Y.to(device)

            labels = torch.nn.functional.one_hot(Y, num_classes=19).float()

            mask =  (1 - (Y == lbl_to_id['PAD_LBL'])).float() # mask is of shape batch_size (batch_size * seq_len)

            optimizer.zero_grad()
            P = net(W, X) #shape = (batch_size, sentence_length, num_tags). This is the P matrix.


            score_curr = ((P * labels).sum(dim=2) * mask).sum(dim=1) #Sanity checked to be correct.
            prob_matrix = net.T

            prob_sum = net.T[Y[:,0:1], lbl_to_id['START_LBL']].squeeze() + (net.T[Y[:,1:], Y[:,:-1]] * mask[:,:-1]).sum(dim=1)

            score = score_curr + prob_sum
            #print("score:", score)


            DP = torch.full((X.shape[0], 19), -1000).cuda() #DP.shape is batch_size * num_tags
            DP[:, lbl_to_id['START_LBL']] = 0.
            for j in range(X.shape[1]):

                sub_mask = mask[:,j].unsqueeze(1)
                DP = (sub_mask) * (log_sum_exp(DP.unsqueeze(1) + prob_matrix.unsqueeze(0) + P[:,j].unsqueeze(2))) + (1 - sub_mask) * DP

            partition = (DP + net.T[lbl_to_id['PAD_LBL']]).logsumexp(dim=1)
            #print("partition:", partition)
            #break
            loss = (partition - score).sum() / X.shape[0] #sum over all minibatches
            loss.backward()


            nn.utils.clip_grad_value_(net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print('[%d, %5d] loss: %f' % (epoch, i + 1, running_loss / 10))
                running_loss = 0.0
        #scheduler.step()
        
        net.eval()
        
        total_loss = 0.0
        for i, mbatch in enumerate(devloader):
            W, X, Y = mbatch
            W, X, Y = W.to(device), X.to(device), Y.to(device)
            
            labels = torch.nn.functional.one_hot(Y, num_classes=19).float()

            mask =  (1 - (Y == lbl_to_id['PAD_LBL'])).float() # mask is of shape batch_size (batch_size * seq_len)

            P = net(W, X) #shape = (batch_size, sentence_length, num_tags). This is the P matrix.


            score_curr = ((P * labels).sum(dim=2) * mask).sum(dim=1) #Sanity checked to be correct.
            prob_matrix = net.T

            prob_sum = net.T[Y[:,0:1], lbl_to_id['START_LBL']].squeeze() + (net.T[Y[:,1:], Y[:,:-1]] * mask[:,:-1]).sum(dim=1)

            score = score_curr + prob_sum


            DP = torch.full((X.shape[0], 19), -1000).cuda() #DP.shape is batch_size * num_tags
            DP[:, lbl_to_id['START_LBL']] = 0.
            for j in range(X.shape[1]):

                sub_mask = mask[:,j].unsqueeze(1)
                DP = (sub_mask) * (log_sum_exp(DP.unsqueeze(1) + prob_matrix.unsqueeze(0) + P[:,j].unsqueeze(2))) + (1 - sub_mask) * DP

            partition = (DP + net.T[lbl_to_id['PAD_LBL']]).logsumexp(dim=1)
            #print("partition:", partition)
            #break
            loss = (partition - score).sum() #sum over all minibatches
            total_loss += loss.item()
            
            
        total_loss = total_loss / len(devset)
        
        print("val loss:", total_loss)
        print("best loss:", min_val_loss)
        print("Patience:", steps_left)
        print("time:", time.time() - tic)
        
        if total_loss < min_val_loss:
            steps_left = patience
            min_val_loss = total_loss
            best_model_dict = net.state_dict()
        else:
            if steps_left == 1:
                early_stop = True
                break
            else:
                steps_left -= 1
    
    print("Early stop:", early_stop)
    if early_stop:
        print("Replacing with better model")
        net.load_state_dict(best_model_dict)
        
    torch.save(net, output_file )
    
    
def predict(saved_model_file, test_set, ner_model, id_to_lbl, lbl_to_id, tok_to_id, pad_lbl_id, output_file):
    
    Y = (1 - (test_set[1] == tok_to_id['PAD_TOK'])).float()
    testset = NERDataset(test_set[0], test_set[1], Y)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = torch.load(saved_model_file).to(device)
    all_labels = []
    for i, mbatch in enumerate(testloader):
        W, X, mask = mbatch
        W, X, mask = W.to(device), X.to(device), mask.to(device)
        P = net(W, X)
        labels = net.predict(P, mask)
        
        for label in labels:
            all_labels.append(label)
    
    return all_labels
        
    
