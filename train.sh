#!/usr/bin/env bash  
#SBATCH -A NAISS2023-5-102 -p alvis # project name, cluster name
#SBATCH -N 1 --gpus-per-node=A40:1     #A40:4 #A100fat:4    #V100:2  A100fat:4  A100:4  # number of nodes, gpu name   
#SBATCH -t 0-11:00:00 # time


#module load AlphaFold/2.1.2-fosscuda-2020b-TensorFlow-2.5.0
#module load AlphaFold/2.2.2-foss-2021a-CUDA-11.3.1
#module load cuDNN/8.2.1.32-CUDA-11.3.1
#module load Seaborn/0.11.2-foss-2021a

source ../../load_modules.sh


python train_ddpm.py