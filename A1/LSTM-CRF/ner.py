#TODO: minimal imports

#TODO: early stopping

#TODO: SGD optimizer

#TODO: lr schedule

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

# load vocab and (glove) embeddings
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

# load classes from (training) data
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

# load data from train/dev/test file
# W: num_seqs x seq_len x word_len - id of characters
# X: num_seqs x seq_len - id of tokens
# Y: num_seqs x seq_len - id of labels
def load_data(path, tok_to_id, lbl_to_id, chr_to_id):
    with open(path, 'r') as f:
        seqs = f.read().split('\n\n')
        seqs.pop()
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

# NER model intended for top-level training
class NERModel(nn.Module):
    def __init__(self, embed_model, seq_tag_model, pad_lbl_id, pad_tok_id):
        super().__init__()
        self.embed_model = embed_model
        self.seq_tag_model = seq_tag_model
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=pad_lbl_id)
        self.pad_lbl_id = pad_lbl_id
        self.pad_tok_id = pad_tok_id

    def forward(self, W, X):
        # trim to max sequence length in batch for speed
        max_len = torch.max(torch.sum(X != self.pad_tok_id, dim=-1))
        X = X[...,:max_len]
        W = W[...,:max_len,:]
        return self.seq_tag_model(self.embed_model(W, X))
    
    def predict(self, W, X):
        with torch.no_grad():
            self.eval()
            Y_hat = self(W, X)
            pred = torch.argmax(Y_hat, dim=-1)
            # restore original shape
            pad = torch.empty(*pred.shape[:-1], X.shape[-1]-pred.shape[-1], dtype=torch.long, device=pred.device).fill_(Y_hat.shape[-1])
            return torch.cat((pred, pad), dim=-1)
        
    def criterion(self, Y, Y_hat):
        # trim to match shape
        Y = Y[...,:Y_hat.shape[-2]]
        return self.cross_entropy_loss(Y_hat.transpose(1,2), Y)
    
    def device(self):
        return next(self.parameters()).device
    
    def batch_predict(self, W, X, batch_size):
        loader = data.DataLoader(data.TensorDataset(W, X), batch_size=batch_size)
        return torch.cat([self.predict(batch_W.to(self.device()), batch_X.to(self.device())) for batch_W, batch_X in loader])
    
# BiLSTM sequence tagger module
class SeqTagModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        # dropout on the input as in lample et. al.
        self.dropout = nn.Dropout(dropout_prob)
        # trainable initial hidden representation and state
        self.h0 = nn.Parameter(torch.randn(2, hidden_size))
        self.c0 = nn.Parameter(torch.randn(2, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, output_size)
    
    def forward(self, X):
        D = self.dropout(X)
        H, _ = self.lstm(D, (self.h0[:,None,:].expand(-1,D.shape[0],-1).contiguous(), self.c0[:,None,:].expand(-1,D.shape[0],-1).contiguous()))
        return self.linear(H)

# LSTMCell module with layer norm
class LSTMCellLN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        k = torch.sqrt(1/torch.tensor(hidden_size, dtype=torch.float))
        self.Wh = nn.Parameter(torch.empty(hidden_size, 4*hidden_size).uniform_(-k, k))
        self.Wx = nn.Parameter(torch.empty(input_size, 4*hidden_size).uniform_(-k, k))
        # initialize forget gate biases to 1 to promote remembering
        self.b = nn.Parameter(torch.cat((torch.ones(hidden_size), torch.zeros(3*hidden_size))))
        self.lnh = nn.LayerNorm(4*hidden_size)
        self.lnx = nn.LayerNorm(4*hidden_size)
        self.lnc = nn.LayerNorm(hidden_size)
    
    def forward(self, X, HC):
        H, C = HC
        f, i, o, g = torch.split(self.lnh(H@self.Wh) + self.lnx(X@self.Wx) + self.b, [self.hidden_size]*4, dim=-1)
        C = torch.sigmoid(f) * C + torch.sigmoid(i) * torch.tanh(g)
        H = torch.sigmoid(o) * torch.tanh(self.lnc(C))
        return H, C

# BiLSTM sequence tagger module based on layer normalized LSTMCell
class LNSeqTagModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm_cell_f = LSTMCellLN(input_size, hidden_size)
        self.h0_f = nn.Parameter(torch.randn(hidden_size))
        self.c0_f = nn.Parameter(torch.randn(hidden_size))
        self.lstm_cell_b = LSTMCellLN(input_size, hidden_size)
        self.h0_b = nn.Parameter(torch.randn(hidden_size))
        self.c0_b = nn.Parameter(torch.randn(hidden_size))
        self.linear = nn.Linear(2*hidden_size, output_size)
    
    def forward(self, X):
        D = self.dropout(X)
        H = torch.empty(X.shape[0], X.shape[1], 2*self.hidden_size, device=X.device)
        
        h = self.h0_f.expand(X.shape[0], -1).contiguous()
        c = self.c0_f.expand(X.shape[0], -1).contiguous()
        for i in range(X.shape[1]):
            h, c = self.lstm_cell_f(D[:,i,:], (h, c))
            H[:,i,:self.hidden_size] = h

        h = self.h0_b.expand(X.shape[0], -1).contiguous()
        c = self.c0_b.expand(X.shape[0], -1).contiguous()
        for i in range(X.shape[1]-1,-1,-1):
            h, c = self.lstm_cell_b(D[:,i,:], (h, c))
            H[:,i,-self.hidden_size:] = h

        return self.linear(H)
    
# token embedding module
class TokEmbModel(nn.Module):
    def __init__(self, init_emb, pad_tok_id):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(init_emb, freeze=False, padding_idx=pad_tok_id)
    
    def forward(self, W, X):
        return self.embedding(X)
    
# character embedding module
class ChrEmbModel(nn.Module):
    def __init__(self, n_embs, pad_chr_id, unk_chr_id, unk_replace_prob, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(n_embs, emb_size, padding_idx=pad_chr_id)
        self.h0 = nn.Parameter(torch.randn(2, hidden_size))
        self.c0 = nn.Parameter(torch.randn(2, hidden_size))
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.unk_chr_id = unk_chr_id
        self.unk_replace_prob = unk_replace_prob

    def forward(self, W):
        X = W.reshape(-1,W.shape[-1])
        if self.training:
            X[torch.empty(X.shape).uniform_() < self.unk_replace_prob] = self.unk_chr_id
        E = self.embedding(X)
        _, (H, _) = self.lstm(E, (self.h0[:,None,:].expand(-1,E.shape[0],-1).contiguous(), self.c0[:,None,:].expand(-1,E.shape[0],-1).contiguous()))
        return H.reshape(*W.shape[:-1],-1)

# character + token embedding module
class ChrTokEmbModel(nn.Module):
    def __init__(self, chr_emb_model, tok_emb_model):
        super().__init__()
        self.chr_emb_model = chr_emb_model
        self.tok_emb_model = tok_emb_model
    
    def forward(self, W, X):
        return torch.cat((self.chr_emb_model(W), self.tok_emb_model(W, X)), dim=-1)

# token level accuracy and macro F1 (note that micro F1 == accuracy at token level)
def metric(Y_true, Y_pred, n_classes):
    with torch.no_grad():
        # trim
        Y_true = Y_true[...,:Y_pred.shape[-1]]
        Y_true = Y_true.reshape(-1)
        Y_pred = Y_pred.reshape(-1)
        # discard pad labels
        Y_pred = Y_pred[Y_true != n_classes]
        Y_true = Y_true[Y_true != n_classes]
        # accuracy
        acc = torch.sum(Y_true == Y_pred).float() / torch.numel(Y_true)
        Z_true = F.one_hot(Y_true, n_classes)
        Z_pred = F.one_hot(Y_pred, n_classes)
        S_tp = torch.sum(Z_true & Z_pred, dim=0).float()
        S_t = torch.sum(Z_true, dim=0).float()
        S_p = torch.sum(Z_pred, dim=0).float()
        f1 = 2 * S_tp / (S_t + S_p)
        # do not consider classes with no examples when taking mean
        f1 = f1[~torch.isnan(f1)]
        macro_F1 = torch.mean(f1)
        return [acc.item(), macro_F1.item()]

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
    plt.suptitle('Loss Curves')

def plot_metrics(train_metrics_np, dev_metrics_np):
    plt.subplot(121)
    plt.plot(train_metrics_np[:,0], label='train')
    plt.plot(dev_metrics_np[:,0], label='dev')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('iterations')
    plt.ylim(0,1)
    plt.subplot(122)
    plt.plot(train_metrics_np[:,1], label='train')
    plt.plot(dev_metrics_np[:,1], label='dev')
    plt.legend()
    plt.title('Macro-F1')
    plt.xlabel('iterations')
    plt.ylim(0,1)
    plt.suptitle('Token-level Metrics at Training')

def train_epoch(model, opt, train_loader, dev_loader, grad_clip_norm, n_classes):
    train_losses = []
    train_metrics = []
    dev_losses = []
    dev_metrics = []
    for (train_W_batch, train_X_batch, train_Y_batch), (dev_W_batch, dev_X_batch, dev_Y_batch) in \
    zip(tqdm(train_loader, 'batches'), it.cycle(dev_loader)):
        with torch.no_grad():
            model.eval()
            dev_W_batch = dev_W_batch.to(model.device())
            dev_X_batch = dev_X_batch.to(model.device())
            dev_Y_batch = dev_Y_batch.to(model.device())
            dev_pred_batch = model(dev_W_batch, dev_X_batch)
            dev_loss = model.criterion(dev_Y_batch, dev_pred_batch)
            dev_losses.append(dev_loss.item())
            dev_metric = metric(dev_Y_batch, torch.argmax(dev_pred_batch, dim=-1), n_classes)
            dev_metrics.append(dev_metric)

        model.train()
        train_W_batch = train_W_batch.to(model.device())
        train_X_batch = train_X_batch.to(model.device())
        train_Y_batch = train_Y_batch.to(model.device())
        train_pred_batch =  model(train_W_batch, train_X_batch)
        train_loss = model.criterion(train_Y_batch, train_pred_batch)
        train_losses.append(train_loss.item())
        train_metric = metric(train_Y_batch, torch.argmax(train_pred_batch, dim=-1), n_classes)
        train_metrics.append(train_metric)
        opt.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        opt.step()
    return train_losses, train_metrics, dev_losses, dev_metrics  

def train_loop(train_set, dev_set, model, opt, n_classes, train_batch_size, dev_batch_size, grad_clip_norm, patience, show):
    train_loader = data.DataLoader(data.TensorDataset(*train_set), batch_size=train_batch_size, shuffle=True)
    dev_loader = data.DataLoader(data.TensorDataset(*dev_set), batch_size=dev_batch_size, shuffle=True)

    train_losses = []
    dev_losses = []
    train_metrics = []
    dev_metrics = []
    
    prev_mean_dev_losses = float('inf')
    while True:
        new_train_losses, new_train_metrics, new_dev_losses, new_dev_metrics = \
        train_epoch(model, opt, train_loader, dev_loader, grad_clip_norm, n_classes)
        
        # early stopping
        mean_dev_losses = sum(new_dev_losses)/len(new_dev_losses)
        if mean_dev_losses < prev_mean_dev_losses:
            patience_ctr = 0
            best = model.state_dict()
        else:
            patience_ctr += 1
            if patience_ctr == patience:
                model.load_state_dict(best)
                break
        prev_mean_dev_losses = mean_dev_losses

        train_losses += new_train_losses
        dev_losses += new_dev_losses
        train_metrics += new_train_metrics
        dev_metrics += new_dev_metrics
        
        if show:
            plt.figure(figsize=(12,4))
            plot_losses(train_losses, dev_losses, new_train_losses, new_dev_losses)
            ipy.display.clear_output(wait=True)
            plt.show()

            train_metrics_np = np.array(train_metrics)
            dev_metrics_np = np.array(dev_metrics)
            plt.figure(figsize=(15,4))
            plot_metrics(train_metrics_np, dev_metrics_np)
            plt.show()
    
    return train_losses, train_metrics, dev_losses, dev_metrics

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
