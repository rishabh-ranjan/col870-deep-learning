import models, train, utils
from importlib import reload
reload(utils)
from torch.utils.data import DataLoader, TensorDataset
import sys
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lenet = models.LeNetDropout()
lenet.load_state_dict(torch.load('data/models/lenet-dropout.pt'))

rrn = models.RRN(24)
rrn.load_state_dict(torch.load('data/models/rrn.pt'))

lenet.eval()
query_X = utils.load_sudoku_images(sys.argv[1], 10000, device, normalize=True)
test_x = utils.decode_sudoku_img(query_X, lenet, device, 256).cpu()
test_X = F.one_hot(test_x, num_classes=9).view(-1,576)

rrn.eval()
rrn.to(device)

predictions = []
loader = DataLoader(TensorDataset(test_X), batch_size=64, shuffle=False)
for i, mbatch in enumerate(loader):
    images = mbatch[0].to(device)
    pred = rrn.predict(images)
    predictions.extend(pred.reshape(-1, 64).tolist())
    
#predictions = rrn.predict(test_X).reshape(-1, 64)

output_file = open(sys.argv[2], 'w')
predictions = torch.tensor(predictions)
for i in range(predictions.shape[0]):
    output_file.write(str(i)+'.png')
    for j in range(64):
        output_file.write(','+str(predictions[i][j].item()))
    
    output_file.write('\n')

output_file.close()
