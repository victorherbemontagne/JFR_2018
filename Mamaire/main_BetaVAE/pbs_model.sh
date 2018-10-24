#!/bin/bash

#PBS -S /bin/bash
#PBS -N JFR
#PBS -j oe
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=24:mem=30gb
#PBS -q gpup100q 
#PBS -P jfr

# Go to the directory where the job has been submitted 
cd $PBS_O_WORKDIR

# Module load
module load anaconda3/5.2

# Environment config
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-9.1/lib64
export CUDA_HOME=/usr/local/cuda-9.1

# Execution
source activate JFR_2018

python main_to_use.py
