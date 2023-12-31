#import libraries
import random
import imageio
import numpy as np
import os
import h5py
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from utils.myDDPM import DDPM, training_loop
from utils.unet import MyBlock, MyUnet
from utils.turbRotDataloader import turbRotDataset

from config import config

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # Unlock the h5py file


#for reproducibility
seed = config.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#dir
parent_dir  = os.getcwd()
parent_dir += '/'
data_dir    = parent_dir + config.data_dir
checkpt_dir = parent_dir + config.checkpt_dir
pred_dir    = parent_dir + config.pred_dir
results_dir = parent_dir + config.results_dir

#definitions
checkpoint  = checkpt_dir + config.checkpt_path

#config
no_train   = config.no_train  
batch_size = config.batch_size
n_epochs   = config.n_epochs
lr         = config.lr 
min_beta   = config.min_beta
max_beta   = config.max_beta
n_steps    = config.n_steps

#Getting Device
device = config.device
print(f"Using device: {device}\t" + (f"Model name: {torch.cuda.get_device_name(0)}"))


dataset = turbRotDataset(data_dir=data_dir)
turbRot_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True )


ddpm = DDPM(MyUnet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
print(f"Number of parameters: {sum([p.numel() for p in ddpm.parameters()])}")


if not no_train:
  training_loop(ddpm, turbRot_dataloader, n_epochs, optim=Adam(ddpm.parameters(), lr), device = device, store_path= checkpoint)


print(f"training is complete")