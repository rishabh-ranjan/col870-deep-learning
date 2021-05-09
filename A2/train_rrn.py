import models, train, utils

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

query_X = torch.load('data/pt-cache/query_X.pt', map_location='cpu')
target_X = torch.load('data/pt-cache/target_X.pt', map_location='cpu')

lenet = models.LeNet()
lenet.load_state_dict(torch.load('data/pt-cache/lenet.pt', map_location='cpu'))

train_x = utils.decode_sudoku_img(query_X, lenet, device, 4096).cpu()
train_X = F.one_hot(train_x, num_classes=9).view(-1,576)
train_y = utils.decode_sudoku_img(target_X, lenet, device, 4096).cpu()

rrn = models.RRN(24)

train.train_rrn(rrn, train_X, train_y, lr=1e-4, batch_size=32, n_epochs='undefined', device=device)

torch.save(rrn.state_dict(), 'data/models/rrn.pt')
