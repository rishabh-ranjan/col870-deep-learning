from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, utils
from tqdm.auto import tqdm

def load_sudoku_images(path, total, device, normalize=False):
    sudoku_img = torch.empty(total,1,224,224, device=device)
    if normalize:
        transform = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))))
    else:
        transform = transforms.ToTensor()
    for i in tqdm(range(total), 'sudoku images'):
        sudoku_img[i,0] = transform(Image.open(f'{path}/{i}.png'))
    return sudoku_img

def balanced_split(X, y, test_size):
    full_X = X.view(-1,1,28,28)
    full_y = y.long()
    class_size = torch.min(torch.bincount(full_y))
    bal_X = torch.cat([full_X[full_y==i][:class_size] for i in range(9)])
    bal_y = torch.cat([full_y[full_y==i][:class_size] for i in range(9)])
    train_X, test_X, train_y, test_y = train_test_split(
        bal_X, bal_y, test_size=test_size, stratify=bal_y)
    return train_X, test_X, train_y, test_y

def split_sudoku_img(sudoku_img):
    return torch.stack(torch.split(
        torch.stack(torch.split(sudoku_img, [28]*8, dim=-2), dim=-3),
        [28]*8, dim=-1), dim=-3).view(-1,1,28,28)

def decode_sudoku_img(sudoku_img, lenet, device, batch_size):
    X = split_sudoku_img(sudoku_img).to(device)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size)
    y = torch.empty(X.shape[0], dtype=torch.long)
    lenet = lenet.to(device)
    for i, [X] in enumerate(tqdm(loader, 'batches')):
        y[i*batch_size:(i+1)*batch_size] = lenet.predict(X)
    return y.view(-1,64)

def arrange_sudoku(img):
    return utils.make_grid(img, nrow=8, padding=0).view(-1,1,224,224)

def viz_images(img, nrow):
    plt.imshow(utils.make_grid(((img+1)/2).detach().cpu(), nrow=nrow, padding=0).permute(1,2,0), cmap='gray')

def store_images(X, path):
    X = X.detach().cpu()
    transform = transforms.ToPILImage()
    for i in tqdm(range(X.shape[0]), 'samples'):
        transform(X[i]).save(os.path.join(path, f'{i}.png'), 'PNG')
        
def constraint_violation(sudoku):
    sudoku = sudoku.view(8,8)
    for i in range(8):
        a = set()
        for j in range(8):
            elem = sudoku[i][j].item()
            a.add(elem)
        if not len(a) == 8:
            return True
        
    for i in range(8):
        a = set()
        for j in range(8):
            elem = sudoku[j][i].item()
            a.add(elem)
        if not len(a) == 8:
            return True
    
    for i in range(0, 8, 2):
        for j in range(0, 8, 4):
            a = set()
            for x in range(0, 2):
                for y in range(0, 4):
                    elem = sudoku[i + x][j + y].item()
                    a.add(elem)
            if not len(a) == 8:
                return True
    return False
