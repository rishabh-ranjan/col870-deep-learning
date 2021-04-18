# test_ner.py --model_file <path to the trained model> --char_embeddings [ 0 | 1 ] --layer_normalization [ 0 | 1 ] --crf [ 0 | 1 ] --test_data_file <path to a file in the same format as original train file with random  NER / POS tags for each token> --output_file <file in the same format as the test data file with random NER tags replaced with the predictions> --glove_embeddings_file <path to file containing glove embeddings> --vocabulary_input_file <path to the vocabulary file written while training>

from ner import *    

import torch

import argparse
import logging

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_file')
parser.add_argument('--initialization')
parser.add_argument('--char_embeddings')
parser.add_argument('--layer_normalization')
parser.add_argument('--crf')
parser.add_argument('--test_data_file')
parser.add_argument('--output_file')
parser.add_argument('--glove_embeddings_file')
parser.add_argument('--vocabulary_input_file')
parser.add_argument('--use_cache', action='store_true')
args = parser.parse_args()

if args.use_cache:
    tok_to_id, glv_emb = torch.load('data/pt-cache/tok_to_id__glv_emb.pt')
else:
    tok_to_id, glv_emb = load_emb(args.glove_embeddings_file, int(4e5))

logging.info('reading token dictionary, character dictionary and class labels from vocabulary file')
tok_to_id, chr_to_id, lbl_to_id, id_to_lbl = torch.load(args.vocabulary_input_file)

if args.use_cache:
    test_W, test_X, _ = torch.load('data/pt-cache/test_W__test_X__test_Y.pt')
else:
    logging.info('loading test set')
    test_W, test_X, _ = load_data(args.test_data_file, tok_to_id, lbl_to_id, chr_to_id)

logging.info('setting device')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'device: {device}')

logging.info('moving to device')
test_W = test_W.to(device)
test_X = test_X.to(device)

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

logging.info('loading model state dict')
ner_model.load_state_dict(torch.load(args.model_file))

logging.info('predicting')
test_pred = ner_model.batch_predict(test_W, test_X, batch_size=2048)

logging.info('writing output file')
test_lbl = to_lbl_seq(test_pred, id_to_lbl)
with open(args.output_file, 'w') as outfile:
    with open(args.test_data_file, 'r') as infile:
        i = -1
        j = 0
        for line in infile:
            if not line.strip():
                print(file=outfile)
                i += 1
                j = 0
                continue
            cols = line.strip().split(' ')
            cols[3] = test_lbl
            print(' '.join(cols), file=outfile)
            j += 1