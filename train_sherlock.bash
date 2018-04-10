#!/bin/bash
#SBATCH --job-name=elman_NN
#SBATCH --time=08:00:00
#SBATCH -p aaiken
#SBATCH --nodes=1
#SBATCH --gres gpu:1


source ${HOME}/setup.bash
cd ${TORCH_RNN_PATH}
source setup.bash
singularity run --nv sherlock-recurrent_nn.simg TRAIN.bash

