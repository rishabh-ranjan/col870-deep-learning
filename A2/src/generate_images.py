import torch
import models, train, utils
import sys
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F

from importlib import reload
reload(utils)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    

def generate_images(gen, device, num_images=5000):
    gen.eval()

    all_labels = torch.empty(0).long()
    all_imgs = torch.empty(0, 1, 28, 28)
    
    for label in tqdm([0,1,2,3,4,5,6,7,8], 'generating images'):
        
        a = torch.zeros(num_images).long() + label
        enc = F.one_hot(a, num_classes=9)
        images = gen(enc.to(device).float())

        all_labels = torch.cat((all_labels, a), dim=0)
        all_imgs = torch.cat((all_imgs, images.detach().cpu()), dim=0)

    return all_imgs, all_labels

    
NUM_IMAGES = 10000
query_X = torch.load('data/pt-cache/query_X_split.pt')
target_X = torch.load('data/pt-cache/target_X_split.pt')
comb_X = torch.cat((query_X, target_X), dim=0)
random_img = comb_X[torch.randint(comb_X.shape[0], (NUM_IMAGES, 1)).flatten()]
utils.store_images(random_img, 'data/images/real')

gen = models.Generator()
gen.load_state_dict(torch.load(sys.argv[3]))
gen.to(device)

img, lbl = generate_images(gen, device, 1000)
utils.store_images(img, 'data/images/fake')

img = img.detach().cpu().numpy().reshape(-1, 784)
lbl = lbl.detach().cpu().numpy().reshape(-1)

img = (img * 128 + 127.5).astype('uint8')

np.save(sys.argv[1], img)
np.save(sys.argv[2], lbl)
