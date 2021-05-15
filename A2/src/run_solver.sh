#!/bin/bash
TRAIN_PATH="$1"
TEST_PATH="$2"
OUTPUT_FILE="$3"
SAMPLE_FILE="$4"
GEN_FILE="$5"
TGT_FILE="$6"

if [ $# -eq 3 ] 
then
    python -u train_rrn.py
    python -u test_rrn.py "$2" "$3"
else
    python -u train_lenet_rrn.py "$4"
    python -u test_lenet_rrn.py "$2" "$7" 
    python -u generate_images.py "$5" "$6" data/models/cgan_gen_adv.pt
    python -u -m pytorch_fid data/images/real data/images/fake
fi
