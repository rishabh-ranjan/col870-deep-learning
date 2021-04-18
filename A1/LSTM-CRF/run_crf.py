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

lbl_to_id = {'B-per': 0, 'I-tim': 1, 'B-org': 2, 'I-per': 3, 'B-tim': 4, 'B-gpe': 5, 'O': 6, 'I-gpe': 7, 'I-nat': 8, 'B-art': 9, 'I-geo': 10, 'I-art': 11, 'I-org': 12, 'B-eve': 13,  'B-geo': 14, 'B-nat': 15, 'I-eve': 16, 'START_LBL': 17, 'PAD_LBL': 18}
id_to_lbl = {0: 'B-per', 1: 'I-tim', 2: 'B-org', 3: 'I-per', 4: 'B-tim', 5: 'B-gpe', 6: 'O', 7: 'I-gpe', 8: 'I-nat', 9: 'B-art', 10: 'I-geo', 11: 'I-art', 12: 'I-org', 13: 'B-eve',  14: 'B-geo', 15: 'B-nat', 16: 'I-eve', 17: 'START_LBL', 18: 'PAD_LBL'}

def load_emb(path, total=None):
    # This functions loads token to id mapping and the embeddings from 'path'
    toks = []
    embs = []
    print("Starting reading embeddings")
    with open(path, 'r') as f:
        for l in f.readlines():
            tok, *emb = l.strip().split()
            emb = [float(x) for x in emb]
            toks.append(tok)
            embs.append(emb)
    print("Embeddings loaded")
    assert('PAD_TOK' not in toks and 'UNK_TOK' not in toks)
    toks += ['PAD_TOK', 'UNK_TOK']
    embs += [[0.]*len(emb), [1.]*len(emb)]
    tok_to_id = dict(zip(toks, it.count()))
    emb = torch.tensor(embs)
    return tok_to_id, emb
            

def load_data(path, tok_to_id, seq_len=128, word_len=64, lim=1e9):
    with open(path, 'r') as f:
        seqs = f.read().split('\n\n')
        seqs.pop()
        seqs[0] = seqs[0][1:]
    X = tok_to_id['PAD_TOK'] * torch.ones((min(len(seqs), lim), seq_len), dtype=torch.long)
    Y = lbl_to_id['PAD_LBL'] * torch.ones((min(len(seqs), lim), seq_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        if i >= lim:
            break
        for j, l in enumerate(seq.split('\n')):
            assert(j < seq_len)
            tok, _, wrd, lbl = l.split(' ')
            try:
                X[i,j] = tok_to_id[tok]
            except KeyError:
                X[i,j] = tok_to_id['UNK_TOK']
                    
            Y[i,j] = lbl_to_id[lbl]
    print("Made sequences")
        
    return X, Y


class NERDataset(data.Dataset):
    def __init__(self, tok_to_id, glv_emb, path='ner-gmb/train.txt', mode='train'):
        global lbl_to_id, id_to_lbl
        '''
        sentences, labels = load_data(path, tok_to_id, lim=1e9)
        self.X = torch.empty((sentences.shape[0], sentences.shape[1], 100))
        self.Y = torch.nn.functional.one_hot(labels, num_classes=19)
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentences[i]):
                self.X[i][j] = glv_emb[sentences[i][j]]
        '''
        self.X = torch.load(mode+'X.pt')
        self.Y = torch.load(mode+'Y.pt')
        print("Loaded embeddings")
        
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

class LinearCRF(nn.Module):
    def __init__(self, input_size, hidden_size, glv_emb, tok_to_id, lstm_model):
        super().__init__()
        
        self.tok_to_id = tok_to_id
        self.tok_emb_model=nn.Embedding.from_pretrained(
            glv_emb,
            freeze=False,
            padding_idx=tok_to_id['PAD_TOK']
        )
        
        #self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.lstm = lstm_model
        
        self.num_tags = 19 #Includes the 17 tags + start + end(PAD_TOK)
        self.T = nn.Parameter(torch.empty((19, 19)))
        #self.T.data[0:17,0:17] = torch.from_numpy(get_matrix().transpose())
        nn.init.normal_(self.T)
        
        
        #17 tag is start and 18 is stop. Arriving on start, beginning on stop is very improbable
        self.T.data[17,:] = -1000 
        self.T.data[:,18] = -1000
        
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Sequential(  nn.Linear(2 * input_size, input_size), 
                                    nn.ReLU(),
                                    nn.Linear(input_size, 19))
        
        
    def forward(self, X):
        #X is of the shape (batch_size, seq_length, num_features)
        
        o, _ = self.lstm(X)
        
        #return F.softmax(self.proj(o), dim=2)
        return self.proj(o)

    def predict(self, P, mask):
        
        with torch.no_grad():
            batch_size = P.shape[0]
            path = []
            for i in range(batch_size):
                path.append([])
                
            choice = torch.zeros(P.shape[0], 128, 19).cuda()
            
            prob_matrix = self.T
            # X.shape is (batch_size, sentence_length)
            
                    
            DP = torch.full((P.shape[0], 19), -1000).cuda()
            DP[:, 17] = 0 #start tag
            for i in range(batch_size):
                for j in range(19):
                    choice[i,0,j] = j
            #path[:,0] = DP[:,0,:].argmax(dim=1)
            
            for i in range(0, 128):
                #next_DP = 
                submask = mask[:, i].unsqueeze(1).float()  # [B, 1]
                emission_score = P[:, i]  # [B, C]

                # [B, 1, C] + [C, C]
                next_choice = DP.unsqueeze(1) + self.T  # [B, C, C]
                next_choice, choice[:, i, :] = next_choice.max(dim=-1)
                next_choice += emission_score
                DP = next_choice * submask + DP * (1 - submask)  # max_score or acc_score_t
                
            DP += self.T[18]
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
                for j in range(num_tags - 2, -1, -1):
                    
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

    
def train():
    
    dir_name = "/home/cse/dual/cs5180404/scratch/col870/crf/results/relu"
    
    os.mkdir(dir_name)
    file_name = os.path.join(dir_name, "log.txt")
    sys.stdout = open(file_name, 'w')
    
    
    #tok_to_id, glv_emb = load_emb('glove/glove.6B.100d.txt', int(4e5))
    tok_to_id, glv_emb = pickle.load(open('tok_to_id', 'rb')), torch.load('glv_emb.pt')
    trainset = NERDataset(tok_to_id, glv_emb, 'ner-gmb/train.txt', 'train')
    devset = NERDataset(tok_to_id, glv_emb, 'ner-gmb/dev.txt', 'dev')
    testset = NERDataset(tok_to_id, glv_emb, 'ner-gmb/test.txt', 'test')
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    devloader = torch.utils.data.DataLoader(devset, batch_size=128, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lstm = nn.LSTM(100, 100, batch_first=True, bidirectional=True)
    net = LinearCRF(100, 100, glv_emb, tok_to_id, lstm).to(device)
    
    print(net)
    
    tic = time.time()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)


    for epoch in range(1, 101):  # loop over the dataset multiple times
        #if epoch % 10 == 0 or epoch == 1:
        net.train()

        running_loss = 0.0
        for i, mbatch in enumerate(trainloader):
            inputs, labels = mbatch
            inputs, labels = inputs.to(device), labels.to(device).float()

            normal_labels = labels.argmax(dim=2) #normal_labels stores the actual labels(not one-hot), shape: (batch_size * seq_len)
            mask =  (1 - (normal_labels == lbl_to_id['PAD_LBL'])).float() # mask is of shape batch_size (batch_size * seq_len)

            optimizer.zero_grad()
            P = net(inputs) #shape = (batch_size, sentence_length, num_tags). This is the P matrix.


            score_curr = ((P * labels).sum(dim=2) * mask).sum(dim=1) #Sanity checked to be correct.
            prob_matrix = net.T

            prob_sum = net.T[normal_labels[:,0:1], lbl_to_id['START_LBL']].squeeze() + (net.T[normal_labels[:,1:], normal_labels[:,:-1]] * mask[:,:-1]).sum(dim=1)

            score = score_curr + prob_sum
            #print("score:", score)


            DP = torch.full((inputs.shape[0], 19), -1000).cuda() #DP.shape is batch_size * num_tags
            DP[:, 17] = 0.
            for j in range(128):

                sub_mask = mask[:,j].unsqueeze(1)
                DP = (sub_mask) * (log_sum_exp(DP.unsqueeze(1) + prob_matrix.unsqueeze(0) + P[:,j].unsqueeze(2))) + (1 - sub_mask) * DP

            partition = (DP + net.T[18]).logsumexp(dim=1)
            #print("partition:", partition)
            #break
            loss = (partition - score).sum() / inputs.shape[0] #sum over all minibatches
            loss.backward()


            nn.utils.clip_grad_value_(net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print('[%d, %5d] loss: %f' % (epoch, i + 1, running_loss / 10))
                running_loss = 0.0
        #scheduler.step()
        
        net.eval()
        
        actual, prediction = [], []
        for i, mbatch in enumerate(devloader):
            inputs, labels = mbatch
            inputs, labels = inputs.to(device), labels.to(device).float()
            normal_labels = labels.argmax(dim=2) #normal_labels stores the actual labels(not one-hot)
            mask =  (1 - (normal_labels == lbl_to_id['PAD_LBL'])) # mask is of shape batch_size (batch_size * seq_len)

            P = net(inputs)
            results = net.predict(P, mask)
            #print(normal_labels)

            if i % 10 == 0:
                print(i)
            for j in range(inputs.shape[0]):
                sgt_actual, sgt_prediction = [], []
                for k in range(128):
                    if mask[j][k]:
                        sgt_actual.append(id_to_lbl[normal_labels[j][k].item()])
                        sgt_prediction.append(id_to_lbl[results[j][k]])
                    else:
                        break

                actual.append(sgt_actual)
                prediction.append(sgt_prediction)

        #print(results)
        print("Val report:\n", classification_report(actual, prediction))
        
        actual, prediction = [], []
        for i, mbatch in enumerate(trainloader):
            inputs, labels = mbatch
            inputs, labels = inputs.to(device), labels.to(device).float()
            normal_labels = labels.argmax(dim=2) #normal_labels stores the actual labels(not one-hot)
            mask =  (1 - (normal_labels == lbl_to_id['PAD_LBL'])) # mask is of shape batch_size (batch_size * seq_len)

            P = net(inputs)
            results = net.predict(P, mask)
            #print(normal_labels)

            if i % 10 == 0:
                print(i)
            for j in range(inputs.shape[0]):
                sgt_actual, sgt_prediction = [], []
                for k in range(128):
                    if mask[j][k]:
                        sgt_actual.append(id_to_lbl[normal_labels[j][k].item()])
                        sgt_prediction.append(id_to_lbl[results[j][k]])
                    else:
                        break

                actual.append(sgt_actual)
                prediction.append(sgt_prediction)
            
        print("Train report:\n", classification_report(actual, prediction))
        
        #print(results)
        #print(classification_report(actual, prediction))
        
        
    torch.save(net, dir_name + '/bilstm-crf.pt' )
if __name__ == '__main__':
    train()