# train_ner.py --initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] --output_file <path to the trained model> --data_dir <directory containing data> --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_output_file <path to the file in which vocabulary will be written>
    
from ner import *    

import torch
import crf

import argparse
import logging
import os

logging.basicConfig(
    format='[ %(asctime)s ] %(message)s',
    level=logging.INFO
)

logging.info('parsing args')
parser = argparse.ArgumentParser()
parser.add_argument('--initialization')
parser.add_argument('--char_embeddings')
parser.add_argument('--layer_normalization')
parser.add_argument('--crf')
parser.add_argument('--output_file')
parser.add_argument('--data_dir')
parser.add_argument('--glove_embeddings_file')
parser.add_argument('--vocabulary_output_file')
parser.add_argument('--cache_dir')
parser.add_argument('--stats_file')
args = parser.parse_args()

logging.info('loading glove embeddings')
cpath = os.path.join(args.cache_dir, 'tok_to_id__glv_emb.pt')
if os.path.isfile(cpath):
    logging.info('\tusing cache')
    tok_to_id, glv_emb = torch.load(cpath)
else:
    logging.info('\tcomputing fresh')
    tok_to_id, glv_emb = load_emb(args.glove_embeddings_file, int(4e5))
    if os.path.isdir(args.cache_dir):
        logging.info('caching')
        torch.save((tok_to_id, glv_emb), cpath)

logging.info('loading character dictionary')
cpath = os.path.join(args.cache_dir, 'chr_to_id.pt')
if os.path.isfile(cpath):
    logging.info('\tusing cache')
    chr_to_id = torch.load(cpath)
else:
    logging.info('\tcomputing fresh')
    chr_to_id = load_chrs(os.path.join(args.data_dir, 'train.txt'))
    if os.path.isdir(args.cache_dir):
        logging.info('caching')
        torch.save(chr_to_id, cpath)

logging.info('loading class labels')
cpath = os.path.join(args.cache_dir, 'lbl_to_id__id_to_lbl.pt')
if os.path.isfile(cpath):
    logging.info('\tusing cache')
    lbl_to_id, id_to_lbl = torch.load(cpath)
else:
    logging.info('\tcomputing fresh')
    lbl_to_id, id_to_lbl = load_classes(os.path.join(args.data_dir, 'train.txt'))
    if os.path.isdir(args.cache_dir):
        logging.info('caching')
        torch.save((lbl_to_id, id_to_lbl), cpath)

logging.info('writing token dictionary, character dictionary and class labels to vocabulary file')
torch.save((tok_to_id, chr_to_id, lbl_to_id, id_to_lbl), args.vocabulary_output_file)

logging.info('loading train set')
cpath = os.path.join(args.cache_dir, 'train_W__train_X__train_Y.pt')
if os.path.isfile(cpath):
    logging.info('\tusing cache')
    train_W, train_X, train_Y = torch.load(cpath)
else:
    logging.info('\tcomputing fresh')
    train_W, train_X, train_Y = load_data(os.path.join(args.data_dir, 'train.txt'), tok_to_id, lbl_to_id, chr_to_id)
    if os.path.isdir(args.cache_dir):
        logging.info('caching')
        torch.save((train_W, train_X, train_Y), cpath)
        
logging.info('loading dev set')
cpath = os.path.join(args.cache_dir, 'dev_W__dev_X__dev_Y.pt')
if os.path.isfile(cpath):
    logging.info('\tusing cache')
    dev_W, dev_X, dev_Y = torch.load(cpath)
else:
    logging.info('\tcomputing fresh')
    dev_W, dev_X, dev_Y = load_data(os.path.join(args.data_dir, 'dev.txt'), tok_to_id, lbl_to_id, chr_to_id)
    if os.path.isdir(args.cache_dir):
        logging.info('caching')
        torch.save((dev_W, dev_X, dev_Y), cpath)

logging.info('setting device')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'device: {device}')

logging.info('moving to device')
train_W = train_W.to(device)
train_X = train_X.to(device)
train_Y = train_Y.to(device)
dev_W = dev_W.to(device)
dev_X = dev_X.to(device)
dev_Y = dev_Y.to(device)

logging.info('creating model')

if args.initialization == 'random':
    init_emb = torch.randn(len(tok_to_id), 100)
elif args.initialization == 'glove':
    init_emb = glv_emb
else:
    assert(False)
    
tok_emb_model = TokEmbModel(
    init_emb=init_emb,
    pad_tok_id=tok_to_id['PAD_TOK']
)

if args.char_embeddings == '0':
    embed_model = tok_emb_model
    emb_size = 100
elif args.char_embeddings == '1':
    embed_model = ChrTokEmbModel(
        chr_emb_model=ChrEmbModel(
            n_embs=len(chr_to_id),
            pad_chr_id=chr_to_id['PAD_CHR'],
            emb_size=16,
            hidden_size=25,
            unk_chr_id=chr_to_id['UNK_CHR'],
            unk_replace_prob=0.2
        ),
        tok_emb_model=tok_emb_model
    )
    emb_size = 150
else:
    assert(False)
    
if args.layer_normalization == '0':
    seq_tag_model = SeqTagModel(
        input_size=emb_size,
        hidden_size=100,
        output_size=len(lbl_to_id) - 1 if args.crf == '0' else len(lbl_to_id) + 1,
        dropout_prob=0.5,
    )
elif args.layer_normalization == '1':
    seq_tag_model = LNSeqTagModel(
        input_size=emb_size,
        hidden_size=100,
        output_size=len(lbl_to_id) - 1 if args.crf == '0' else len(lbl_to_id) + 1,
        dropout_prob=0.5
    )
else:
    assert(False)
    
# TODO: CRF
if args.crf == '0':
    ner_model = NERModel(
        embed_model=embed_model,
        seq_tag_model=seq_tag_model,
        pad_lbl_id=lbl_to_id['PAD_LBL'],
        pad_tok_id=tok_to_id['PAD_TOK']
    )

    logging.info('moving model to device')
    ner_model = ner_model.to(device)

    logging.info('training')

    stats = train_loop(
        train_set=(train_W, train_X, train_Y),
        dev_set=(dev_W, dev_X, dev_Y),
        model=ner_model,
        lr=1e-3,
        cos_max=100,
        n_classes=len(lbl_to_id)-1,
        train_batch_size=128,
        dev_batch_size=128,
        grad_clip_norm=5,
        patience=5,
        max_epochs=100,
        show=False, 
        id_to_lbl=id_to_lbl, 
        pad_lbl_id=lbl_to_id['PAD_LBL']
    )

    logging.info('saving model state dict')
    torch.save(ner_model.state_dict(), args.output_file)

    if args.stats_file:
        logging.info('saving stats')
        torch.save(stats, args.stats_file)

else:
    logging.info('trying crf')
    ner_model = NERModel(
        embed_model=embed_model,
        seq_tag_model=seq_tag_model,
        pad_lbl_id=lbl_to_id['PAD_LBL'],
        pad_tok_id=tok_to_id['PAD_TOK']
    )

    logging.info('moving model to device')
    ner_model = ner_model.to(device)

    logging.info('training')
    id_to_lbl.append('START_LBL')
    lbl_to_id['START_LBL'] = len(id_to_lbl) - 1
    
    print(id_to_lbl, lbl_to_id)
    crf.train( train_set=(train_W.cpu(), train_X.cpu(), train_Y.cpu()),
                dev_set=(dev_W.cpu(), dev_X.cpu(), dev_Y.cpu()),
                ner_model=ner_model,
                id_to_lbl=id_to_lbl, 
                lbl_to_id=lbl_to_id,
                pad_lbl_id=lbl_to_id['PAD_LBL'],
                output_file=args.output_file,
                
             )
