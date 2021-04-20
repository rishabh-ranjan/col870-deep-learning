#!/bin/sh
#PBS -N job_ner
#PBS -P cse
#PBS -lselect=1:ncpus=1:ngpus=1
#PBS -lwalltime=08:00:00


cd /home/cse/dual/cs5180404/scratch/col870/crf/
module load apps/anaconda/3
python train_ner.py --initialization glove --char_embeddings 1 --layer_normalization 1 --crf 1 --output_file models/crf_111.pt --data_dir ner-gmb/ --glove_embeddings glove/glove.6B.100d.txt --vocabulary_output_file vocab/crf_111

