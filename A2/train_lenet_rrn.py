import models, train

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

query_X = torch.load('data/pt-cache/query_X.pt', map_location='cpu')
target_X = torch.load('data/pt-cache/target_X.pt', map_location='cpu')

samples = torch.load('data/pt-cache/samples.pt', map_location='cpu')

net = models.LeNetRRN(24, 1, samples)

net.lenet.load_state_dict(torch.load('data/pt-cache/lenet.pt', map_location='cpu'))
net.rrn.load_state_dict(torch.load('data/pt-cache/rrn.pt', map_location='cpu'))

train.train_net(net, query_X, target_X, lr=1e-3, batch_size=32, n_epochs=20, device=device)

torch.save(net.state_dict(), 'data/models/lenet_rrn.pt')
torch.save(net.lenet.state_dict(), 'data/models/lenet_adv.pt')
torch.save(net.rrn.state_dict(), 'data/models/rrn_adv.pt')
