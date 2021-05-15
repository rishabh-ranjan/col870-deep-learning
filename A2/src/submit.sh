#!/bin/sh


#PBS -P cse
#PBS -N job_gan
#PBS -lselect=1:ncpus=1:ngpus=1:centos=skylake
#PBS -lwalltime=06:00:00

echo "==============================="
#echo $PBS_JOBID
#cat $PBS_NODEFILE
echo "==============================="

cd /home/cse/dual/cs5180404/scratch/col870/sudoku_sub/
module load apps/anaconda/3

source /home/cse/dual/cs5180404/scratch/col870/sudoku_sub/run_generator.sh ~/scratch/col870/a2/data/visual_sudoku/train ~/scratch/col870/a2/data/sample_images.npy gen9k.npy target9k.npy
echo "---------------------------------------------------------------------------------------------------------------------------------------"


source /home/cse/dual/cs5180404/scratch/col870/sudoku_sub/run_solver.sh /home/cse/dual/cs5180404/scratch/col870/a2/data/visual_sudoku/train /home/cse/dual/cs5180404/scratch/col870/a2/data/visual_sudoku/test output.csv 
echo "---------------------------------------------------------------------------------------------------------------------------------------"

source /home/cse/dual/cs5180404/scratch/col870/sudoku_sub/run_solver.sh /home/cse/dual/cs5180404/scratch/col870/a2/data/visual_sudoku/train /home/cse/dual/cs5180404/scratch/col870/a2/data/visual_sudoku/test joint_output.csv /home/cse/dual/cs5180404/scratch/col870/a2/data/sample_images.npy gen1k_joint.npy target1k_joint.npy output_joint.csv
echo "---------------------------------------------------------------------------------------------------------------------------------------"
