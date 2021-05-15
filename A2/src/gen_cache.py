import sys
import os
import utils
import torch
import logging
import numpy as np

logging.basicConfig(
    format='[ %(asctime)s ] %(message)s',
    level=logging.INFO
) 


if __name__ == '__main__':
    
    
    train_path = sys.argv[1]
    query_path = os.path.join(train_path, 'query')
    target_path = os.path.join(train_path, 'target')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')
    
    data_folder = 'data/'
    
    if not os.path.exists(data_folder):
        logging.info('Created data folder')
        os.mkdir(data_folder)
    
    # The data folder exists now
    
    
    pt_cache_folder = os.path.join(data_folder, 'pt-cache')
    
    if not os.path.exists(pt_cache_folder):
        logging.info('Created cache folder')
        os.mkdir(pt_cache_folder)
    
    # The pt_cache exists now
    
    dataset_folder = os.path.join(data_folder, 'dataset')
    
    if not os.path.exists(dataset_folder):
        logging.info('Created dataset folder')
        os.mkdir(dataset_folder)
        
        
    model_folder = os.path.join(data_folder, 'models')
    
    if not os.path.exists(model_folder):
        logging.info('Created model folder')
        os.mkdir(model_folder)
        
    img_folder = os.path.join(data_folder, 'images')
    
    if not os.path.exists(img_folder):
        logging.info('Created image folder')
        os.mkdir(img_folder)
        os.mkdir(os.path.join(img_folder, 'real'))
        os.mkdir(os.path.join(img_folder, 'fake'))
    
    
    query_cache = os.path.join(pt_cache_folder, 'query_X_split.pt')
    target_cache = os.path.join(pt_cache_folder, 'target_X_split.pt') 
    samples_cache = os.path.join(pt_cache_folder, 'samples.pt')
                    
    query_found = os.path.exists(query_cache)
    target_found = os.path.exists(target_cache)
    samples_found = os.path.exists(samples_cache)
    #Add other other caches if needed
    
    
    if not query_found:
        logging.info('Generating query cache')
        query_sudokus = utils.load_sudoku_images(query_path, 10000, device, normalize=True)
        torch.save(query_sudokus.cpu(), os.path.join(pt_cache_folder, 'query_X.pt'))
        query_digits = utils.split_sudoku_img(query_sudokus)
        torch.save(query_digits.cpu(), query_cache)
    else:
        logging.info('Query cache exists')
    
    if not target_found:
        logging.info('Generating target cache')
        target_sudokus = utils.load_sudoku_images(target_path, 10000, device, normalize=True)
        torch.save(target_sudokus.cpu(), os.path.join(pt_cache_folder, 'target_X.pt'))
        target_digits = utils.split_sudoku_img(target_sudokus)
        torch.save(target_digits.cpu(), target_cache)
    else:
        logging.info('Target cache exists')
    
    if not samples_found:
        samples_path = sys.argv[2]
        logging.info('Generating samples cache')
        samples = torch.tensor((np.load(samples_path))[0:9])[:,None,:,:]
        samples = (samples.float() - 127.5) / 255.0
        torch.save(samples.cpu(), samples_cache)
    else:
        logging.info('Samples cache exists')
    
        
    
    
    
    
    