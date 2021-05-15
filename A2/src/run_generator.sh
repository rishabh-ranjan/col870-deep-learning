#!/bin/bash
TRAIN_PATH="$1"
SAMPLE_PATH="$2"
IMG_PATH="$3"
LBL_PATH="$4"

python -u gen_cache.py $TRAIN_PATH $SAMPLE_PATH
python -u train_cgan_dropout.py $SAMPLE_PATH $IMG_PTH $LBL_PTH
python -u generate_images.py $IMG_PATH $LBL_PATH data/models/cgan_gen.pt
python -u -m pytorch_fid data/images/real data/images/fake
