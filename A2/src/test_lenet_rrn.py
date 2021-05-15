import models, train, utils
from importlib import reload
reload(utils)
from torch.utils.data import DataLoader, TensorDataset
import sys
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

joint_rrn = models.LeNetRRN(24, 0.5, 0.5)
joint_rrn.load_state_dict(torch.load('data/models/lenet_rrn.pt'))
joint_rrn.to(device)
joint_rrn.eval()

query_X = utils.load_sudoku_images(sys.argv[1], 10000, device, normalize=True)
query_X.to(device)


predictions = []
loader = DataLoader(TensorDataset(query_X), batch_size=64, shuffle=False)
for i, mbatch in enumerate(loader):
    images = mbatch[0].to(device)
    pred = joint_rrn.predict(images)
    predictions.extend(pred.reshape(-1, 64).tolist())
    
output_file = open(sys.argv[2], 'w')
predictions = torch.tensor(predictions)

for i in range(predictions.shape[0]):
    output_file.write(str(i)+'.png')
    for j in range(64):
        output_file.write(','+str(predictions[i][j].item()))
    
    output_file.write('\n')

output_file.close()

