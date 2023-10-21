#import libraries
import random
import imageio
import numpy as np
import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

from utils.myDDPM import DDPM, training_loop
from utils.unet import MyBlock, MyUnet

from config import config

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"


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
fashion    = config.fashion   
batch_size = config.batch_size
n_epochs   = config.n_epochs
lr         = config.lr 
min_beta   = config.min_beta
max_beta   = config.max_beta
n_steps    = config.n_steps

#Getting Device
device = config.device
print(f"Using device: {device}\t" + (f"Model name: {torch.cuda.get_device_name(0)}"))


# converting images to tensor and normalizing

transform = Compose([                                    #Compose class is typically used to define pipelines to perfrom data processing (mostly for images)
    ToTensor(),                                          #ToTensor() convert image to pytorch tensor in (Channel, Height, width)
    Lambda(lambda x: (x - 0.5)*2)                         #Custom lambda function: substracts 0.5 from each element and multiplies by 2. To normalize the data between [-1,1]
])


ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn(data_dir, download=False, train=True, transform=transform)    #when downloading data from torchvision, download=True (download if already not there), train=true(perform train test split), transform (aply the transform pipeline)
loader = DataLoader(dataset, batch_size, shuffle=True)

ddpm = DDPM(MyUnet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

print(f"Number of parameters: {sum([p.numel() for p in ddpm.parameters()])}")


if not no_train:
  training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device = device, store_path= checkpoint)


print(f"training is complete")