import IPython as ipy
import matplotlib.pyplot as plt
import numpy as np
from seqeval.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.optim as optim
import torch.utils.data as data
from tqdm.auto import tqdm
import itertools as it
import time

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

def load_data(path, tok_to_id, lbl_to_id, chr_to_id, seq_len=128, word_len=64):
    with open(path, 'r') as f:
        seqs = f.read().split('\n\n')
        seqs.pop()
        seqs[0] = seqs[0][1:]
    X = tok_to_id['PAD_TOK'] * torch.ones((len(seqs), seq_len), dtype=torch.long)
    Y = lbl_to_id['PAD_LBL'] * torch.ones((len(seqs), seq_len), dtype=torch.long)
    W = chr_to_id['PAD_CHR'] * torch.ones((len(seqs), seq_len, word_len), dtype=torch.long)
    for i, seq in enumerate(tqdm(seqs, 'sequences')):
        for j, l in enumerate(seq.split('\n')):
            assert(j < seq_len)
            tok, _, wrd, lbl = l.split(' ')
            try:
                X[i,j] = tok_to_id[tok]
            except KeyError:
                X[i,j] = tok_to_id['UNK_TOK']
                
            for k, ch in enumerate(wrd):
                try:
                    W[i,j,k] = chr_to_id[ch]
                except KeyError:
                    W[i,j,k] = chr_to_id['UNK_CHR']
                    
            Y[i,j] = lbl_to_id[lbl]
    return X, Y, W

class NERModel(nn.Module):
    def __init__(self, embed_model, seq_tag_model, pad_lbl_id, pad_tok_id, class_freq):
        super().__init__()
        self.embed_model = embed_model
        self.seq_tag_model = seq_tag_model
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=1/class_freq.float(), ignore_index=pad_lbl_id)
        self.pad_lbl_id = pad_lbl_id
        self.pad_tok_id = pad_tok_id

    def forward(self, W, X):
        max_len = torch.max(torch.sum(X != self.pad_tok_id, dim=-1))
        X = X[...,:max_len]
        W = W[...,:max_len,:]
        return self.seq_tag_model(self.embed_model(W, X))
    
    def predict(self, W, X):
        with torch.no_grad():
            self.eval()
            Y_hat = self(W, X)
            pred = torch.argmax(Y_hat, dim=-1)
            return torch.cat((pred, Y_hat.shape[-1] * torch.ones(*pred.shape[:-1], X.shape[-1]-pred.shape[-1], dtype=torch.long, device=pred.device)), dim=-1)
        
    def criterion(self, Y, Y_hat):
        Y = Y[...,:Y_hat.shape[-2]]
        return self.cross_entropy_loss(Y_hat.transpose(1,2), Y)
    
    def device(self):
        return next(self.parameters()).device
    
    def batch_predict(self, W, X, batch_size):
        loader = data.DataLoader(data.TensorDataset(W, X), batch_size=batch_size)
        return torch.cat([self.predict(batch_W.to(self.device()), batch_X.to(self.device())) for batch_W, batch_X in loader])
    
class SeqTagModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.h0 = nn.Parameter(torch.zeros(2, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(2, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, output_size)
    
    def forward(self, X):
        D = self.dropout(X)
        H, _ = self.lstm(D, (self.h0[:,None,:].expand(-1,D.shape[0],-1).contiguous(), self.c0[:,None,:].expand(-1,D.shape[0],-1).contiguous()))
        return self.linear(H)

class LNSeqTagModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm_cell_f = nn.LSTMCell(input_size, hidden_size)
        self.h0_f = nn.Parameter(torch.zeros(hidden_size))
        self.c0_f = nn.Parameter(torch.zeros(hidden_size))
        self.lstm_cell_b = nn.LSTMCell(input_size, hidden_size)
        self.h0_b = nn.Parameter(torch.zeros(hidden_size))
        self.c0_b = nn.Parameter(torch.zeros(hidden_size))
        self.linear = nn.Linear(2*hidden_size, output_size)
    
    def forward(self, X):
        D = self.dropout(X)
        H = torch.empty(X.shape[0], X.shape[1], 2*self.hidden_dims, device=X.device)
        
        h = self.h0_f.expand(X.shape[0], -1).contiguous()
        c = self.c0_f.expand(X.shape[0], -1).contiguous()
        for i in range(X.shape[1]):
            h, c = self.lstm_cell_f(D[:,i,:], (h, c))
            H[:,i,:self.hidden_dims] = h

        h = self.h0_b.expand(X.shape[0], -1).contiguous()
        c = self.c0_b.expand(X.shape[0], -1).contiguous()
        for i in range(X.shape[1]-1,-1,-1):
            h, c = self.lstm_cell_b(D[:,i,:], (h, c))
            H[:,i,-self.hidden_dims:] = h

        return self.linear(H)
    
class TokEmbModel(nn.Module):
    def __init__(self, init_emb, pad_tok_id):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(init_emb, freeze=False, padding_idx=pad_tok_id)
    
    def forward(self, W, X):
        return self.embedding(X)
    
class ChrEmbModel(nn.Module):
    def __init__(self, n_embs, pad_chr_id, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(n_embs, emb_size, padding_idx=pad_chr_id)
        self.h0 = nn.Parameter(torch.zeros(2, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(2, hidden_size))
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, W):
        X = W.reshape(-1,W.shape[-1])
        E = self.embedding(X)
        _, (H, _) = self.lstm(E, (self.h0[:,None,:].expand(-1,E.shape[0],-1).contiguous(), self.c0[:,None,:].expand(-1,E.shape[0],-1).contiguous()))
        return H.reshape(*W.shape[:-1],-1)

class ChrTokEmbModel(nn.Module):
    def __init__(self, chr_emb_model, tok_emb_model):
        super().__init__()
        self.chr_emb_model = chr_emb_model
        self.tok_emb_model = tok_emb_model
    
    def forward(self, W, X):
        return torch.cat((self.chr_emb_model(W), self.tok_emb_model(X)), dim=-1)
    
def metrics(Y_true, Y_pred, n_classes):
    with torch.no_grad():
        assert(Y_true.shape == Y_pred.shape)
        Y_true = Y_true.reshape(-1)
        Y_pred = Y_pred.reshape(-1)
        Y_pred = Y_pred[Y_true != n_classes]
        Y_true = Y_true[Y_true != n_classes]
        acc = torch.sum(Y_true == Y_pred).float() / torch.numel(Y_true)
        Z_true = F.one_hot(Y_true, n_classes)
        Z_pred = F.one_hot(Y_pred, n_classes)
        S_tp = torch.sum(Z_true & Z_pred, dim=0).float()
        S_t = torch.sum(Z_true, dim=0).float()
        S_p = torch.sum(Z_pred, dim=0).float()
        micro_F1 = 2 * torch.sum(S_tp) / (torch.sum(S_t) + torch.sum(S_p))
        macro_F1 = torch.mean(2 * S_tp / (S_t + S_p))
        return [acc.item(), micro_F1.item(), macro_F1.item()]

def plot_losses(train_losses, dev_losses, new_train_losses, new_dev_losses):
    plt.subplot(121)
    plt.plot(dev_losses, label='dev')
    plt.plot(train_losses, label='train')
    plt.legend()
    plt.title('All Epochs')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.plot(new_dev_losses, label='dev')
    plt.plot(new_train_losses, label='train')
    plt.legend()
    plt.title('Last Epoch')
    plt.xlabel('iterations')
    plt.ylabel('loss')

def plot_metrics(train_metrics_np, dev_metrics_np):
    plt.subplot(131)
    plt.plot(train_metrics_np[:,0], 'x', label='train')
    plt.plot(dev_metrics_np[:,0], 'x', label='dev')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylim(0,1)
    plt.subplot(132)
    plt.plot(train_metrics_np[:,1], 'x', label='train')
    plt.plot(dev_metrics_np[:,1], 'x', label='dev')
    plt.legend()
    plt.suptitle('Micro-F1')
    plt.xlabel('epochs')
    plt.ylim(0,1)
    plt.subplot(133)
    plt.plot(train_metrics_np[:,2], 'x', label='train')
    plt.plot(dev_metrics_np[:,2], 'x', label='dev')
    plt.legend()
    plt.title('Macro-F1')
    plt.xlabel('epochs')
    plt.ylim(0,1)

def train_epoch(model, opt, train_loader, dev_loader, grad_clip_norm):
    train_losses = []
    dev_losses = []
    for (train_W_batch, train_X_batch, train_Y_batch), (dev_W_batch, dev_X_batch, dev_Y_batch) in \
    zip(tqdm(train_loader, 'batches'), it.cycle(dev_loader)):
        with torch.no_grad():
            model.eval()
            dev_W_batch = dev_W_batch.to(model.device())
            dev_X_batch = dev_X_batch.to(model.device())
            dev_Y_batch = dev_Y_batch.to(model.device())
            dev_loss = model.criterion(dev_Y_batch, model(dev_W_batch, dev_X_batch))
            dev_losses.append(dev_loss.item())

        model.train()
        train_W_batch = train_W_batch.to(model.device())
        train_X_batch = train_X_batch.to(model.device())
        train_Y_batch = train_Y_batch.to(model.device())
        train_loss = model.criterion(train_Y_batch, model(train_W_batch, train_X_batch))
        train_losses.append(train_loss.item())
        opt.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()
    return train_losses, dev_losses    

def train_loop(train_set, dev_set, model, opt, n_classes, batch_size, grad_clip_norm, pred_batch_size):
    train_W, train_X, train_Y = train_set
    dev_W, dev_X, dev_Y = dev_set
    train_losses = []
    dev_losses = []
    train_metrics = []
    dev_metrics = []
    train_loader = data.DataLoader(data.TensorDataset(*train_set), batch_size=batch_size, shuffle=True)
    dev_loader = data.DataLoader(data.TensorDataset(*dev_set), batch_size=batch_size, shuffle=True)
    while True:
        new_train_losses, new_dev_losses = train_epoch(model, opt, train_loader, dev_loader, grad_clip_norm)
        
        train_losses += new_train_losses
        dev_losses += new_dev_losses
        plt.figure(figsize=(12,4))
        plot_losses(train_losses, dev_losses, new_train_losses, new_dev_losses)
        ipy.display.clear_output(wait=True)
        plt.show()
        
        new_train_metrics = metrics(train_Y.to(model.device()), model.batch_predict(train_W, train_X, pred_batch_size), n_classes)
        train_metrics.append(new_train_metrics)
        new_dev_metrics = metrics(dev_Y.to(model.device()), model.batch_predict(dev_W, dev_X, pred_batch_size), n_classes)
        dev_metrics.append(new_dev_metrics)
        train_metrics_np = np.array(train_metrics)
        dev_metrics_np = np.array(dev_metrics)
        plt.figure(figsize=(15,4))
        plot_metrics(train_metrics_np, dev_metrics_np)
        plt.show()
        
        print(new_train_losses[-1])
        print(new_dev_losses[-1])

def to_lbl_seq(Y, id_to_lbl):
    ret = []
    for seq in Y:
        row = []
        for lbl_id in seq:
            lbl = id_to_lbl[lbl_id]
            if lbl == 'PAD_LBL':
                break
            row.append(lbl)
        ret.append(row)
    return ret

def conll_report(Y_true, Y_pred, id_to_lbl, pad_lbl_id):
    assert(Y_true.shape == Y_pred.shape)
    Y_pred[Y_true == pad_lbl_id] = pad_lbl_id
    return classification_report(to_lbl_seq(Y_true, id_to_lbl), to_lbl_seq(Y_pred, id_to_lbl))
