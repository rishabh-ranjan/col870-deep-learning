# train_ner.py --initialization [random | glove ] --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] --output_file <path to the trained model> --data_dir <directory containing data> --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_output_file <path to the file in which vocabulary will be written>
    
from ner import *    

import torch

import argparse
import logging
import os

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
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
parser.add_argument('--use_cache', action='store_true')
args = parser.parse_args()

if args.use_cache:
    tok_to_id, glv_emb = torch.load('data/pt-cache/tok_to_id__glv_emb.pt')
    chr_to_id = torch.load('data/pt-cache/chr_to_id.pt')
    lbl_to_id, id_to_lbl = torch.load('data/pt-cache/lbl_to_id__id_to_lbl')
    train_W, train_X, train_Y = torch.load('data/pt-cache/train_W__train_X__train_Y.pt')
    dev_W, dev_X, dev_Y = torch.load('data/pt-cache/dev_W__dev_X__dev_Y.pt')
else:
    logging.info('loading glove embeddings')
    tok_to_id, glv_emb = load_emb(args.glove_embeddings_file, int(4e5))

    logging.info('loading character dictionary')
    chr_to_id = load_chrs(os.path.join(args.data_dir, 'train.txt'))

    logging.info('loading class labels')
    lbl_to_id, id_to_lbl = load_classes(os.path.join(args.data_dir, 'train.txt'))

    logging.info('writing token dictionary, character dictionary and class labels to vocabulary file')
    torch.save((tok_to_id, chr_to_id, lbl_to_id, id_to_lbl), args.vocabulary_output_file)

    logging.info('loading train set')
    train_W, train_X, train_Y = load_data(os.path.join(args.data_dir, 'train.txt'), tok_to_id, lbl_to_id, chr_to_id)

    logging.info('loading dev set')
    dev_W, dev_X, dev_Y = load_data(os.path.join(args.data_dir, 'dev.txt'), tok_to_id, lbl_to_id, chr_to_id)

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
    assert(false)
    
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
    assert(false)
    
if args.layer_normalization == '0':
    seq_tag_model = SeqTagModel(
        input_size=emb_size,
        hidden_size=100,
        output_size=len(lbl_to_id)-1,
        dropout_prob=0.5,
    )
elif args.layer_normalization == '1':
    seq_tag_model = LNSeqTagModel(
        input_size=emb_size,
        hidden_size=100,
        output_size=len(lbl_to_id)-1,
        dropout_prob=0.5
    )
else:
    assert(false)
    
# TODO: CRF

ner_model = NERModel(
    embed_model=embed_model,
    seq_tag_model=seq_tag_model,
    pad_lbl_id=lbl_to_id['PAD_LBL'],
    pad_tok_id=tok_to_id['PAD_TOK']
)

logging.info('moving model to device')
ner_model = ner_model.to(device)

logging.info('training')

train_loop(
    train_set=(train_W, train_X, train_Y),
    dev_set=(dev_W, dev_X, dev_Y),
    model=ner_model,
    opt=optim.Adam(ner_model.parameters(), lr=0.1),
    n_classes=len(lbl_to_id)-1,
    train_batch_size=128,
    dev_batch_size=128,
    grad_clip_norm=5,
    patience=2,
    show=False
)

logging.info('saving model state dict')
torch.save(ner_model.state_dict(), args.output_file)