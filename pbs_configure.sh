#!/bin/bash

# Module load
module load anaconda3/5.2

# Environment config
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-9.1/lib64
export CUDA_HOME=/usr/local/cuda-9.1
conda env create -f environment.yml --force
conda install -c conda-forge nibabel
conda install tqdm

# Execution
source activate JFR

python main.py