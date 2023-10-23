import torch
import numpy as np
import os

from utils.myDDPM import DDPM
from utils.unet import MyBlock, MyUnet
from utils.post_proc_images import show_images, generate_new_images
from utils.ddpm_videos import animationDDPM

from config import config

os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"



#dir
parent_dir  = os.getcwd()
parent_dir += '/'
data_dir    = parent_dir + config.data_dir
checkpt_dir = parent_dir + config.checkpt_dir
pred_dir    = parent_dir + config.pred_dir
results_dir = parent_dir + config.results_dir


#definitions
checkpoint  = checkpt_dir + config.checkpt_path
video_path  = results_dir + config.video_path

#config
#no_train   = config.no_train 
#batch_size = config.batch_size
#n_epochs   = config.n_epochs
#lr         = config.lr
#min_beta   = config.min_beta
#max_beta   = config.max_beta
n_samples  = config.n_samples
n_steps    = config.n_steps 

#Getting Device
device = config.device
print(f"Using device: {device}\t" + (f"Model name: {torch.cuda.get_device_name(0)}"))


# Load trained model 

best_model = DDPM(MyUnet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(checkpoint, map_location=device))
best_model.eval()
print("Model loaded")


print("Generating new images")
generated, frames = generate_new_images( 
        best_model,
        n_samples=100,
        device=device,
    )

#grid size
x = np.linspace(0,639, 640)
y = np.linspace(0,639, 640)


animation = animationDDPM(frames, video_path, fps = 4, x=x, y=y)
animation.create_video()

print("saved animation")