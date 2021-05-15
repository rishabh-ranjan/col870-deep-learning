import models, train, utils
from importlib import reload
reload(models)
reload(train)
reload(utils)

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

query_X = torch.load('data/pt-cache/query_X.pt', map_location='cpu')
target_X = torch.load('data/pt-cache/target_X.pt', map_location='cpu')

lenet = models.LeNetDropout()
lenet.load_state_dict(torch.load('data/models/lenet-dropout.pt', map_location='cpu'))

train_x = utils.decode_sudoku_img(query_X, lenet, device, 256).cpu()
train_X = F.one_hot(train_x, num_classes=9).view(-1,576)
train_y = utils.decode_sudoku_img(target_X, lenet, device, 256).cpu()

rrn = models.RRN(24)

train_split_X, val_split_X = torch.split(train_X, [9500,500], dim=0)
train_split_y, val_split_y = torch.split(train_y, [9500,500], dim=0)

train.train_rrn_val(rrn, train_split_X, train_split_y, val_split_X, val_split_y, lr=4e-4, batch_size=64, n_epochs=125, device=device, steps=[0,1,3,7,15,23], show_step=282)

torch.save(rrn.state_dict(), 'data/models/rrn.pt')
