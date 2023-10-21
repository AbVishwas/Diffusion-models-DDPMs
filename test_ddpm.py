import torch
from utils.myDDPM import DDPM
from utils.unet import MyBlock, MyUnet
import os
from utils.post_proc_images import show_images, generate_new_images
from utils.ddpm_videos import animationDDPM
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

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


# Load trained model 

best_model = DDPM(MyUnet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(checkpoint, map_location=device))
best_model.eval()
print("Model loaded")


print("Generating new images")
generated, frames = generate_new_images(    #, frames
        best_model,
        n_samples=100,
        device=device,
    )


x = np.linspace(0,279, 280)
y = np.linspace(0,279, 280)


animation = animationDDPM(frames, results + "frames.mp4", fps = 2, x=x, y=y)
animation.create_video()

print("saved animation")