#import libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from utils.myDDPM import DDPM, training_loop
from utils.unet import MyBlock, MyUnet

import os

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"


#for reproducibility
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#dir
parent_dir = os.getcwd()
parent_dir += '/' 
data_dir = parent_dir + "01_Data/"
checkpt_dir = parent_dir + "02_checkpoints/"
pred_dir = parent_dir + "03_preds/"
results = parent_dir + "04_results/"

#definitions
checkpoint = checkpt_dir + "ddpm_model_fashion.pt"

#config
no_train = False
fashion = True #weather to use fashion MNIST or normal MNIST
batch_size = 128
n_epochs = 20
lr = 0.001
n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors

#Getting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"Model name: {torch.cuda.get_device_name(0)}"))



# converting images to tensor and normalizing

transform = Compose([                                    #Compose class is typically used to define pipelines to perfrom data processing (mostly for images)
    ToTensor(),                                          #ToTensor() convert image to pytorch tensor in (Channel, Height, width)
    Lambda(lambda x: (x - 0.5)*2)                         #Custom lambda function: substracts 0.5 from each element and multiplies by 2. To normalize the data between [-1,1]
])


ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn(data_dir, download=False, train=True, transform=transform)    #when downloading data from torchvision, download=True (download if already not there), train=true(perform train test split), transform (aply the transform pipeline)
loader = DataLoader(dataset, batch_size, shuffle=True)


#Getting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"Model name: {torch.cuda.get_device_name(0)}"))



ddpm = DDPM(MyUnet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)


print(f"Number of parameters: {sum([p.numel() for p in ddpm.parameters()])}")



if not no_train:
  training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device = device, store_path= checkpoint)


print(f"training is complete")