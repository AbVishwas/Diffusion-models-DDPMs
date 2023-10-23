import torch
import torch.nn as nn

class config:

    """Devices""" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Datasets"""
    dataset = "turbRot"

    if dataset == "Fashion":  
        h,w = 28,28
        
        """Directories"""

        data_dir    = "01_Data/FashionMNIST"
        checkpt_dir = "02_checkpoints/"
        pred_dir    = "03_preds/"
        results_dir = "04_results/" 
        
        """Results_paths"""
        checkpt_path    =  "ddpm_model_fashion.pt"
        video_path      =  "ddpm_fashion.mp4"

        """Training and Testing parameters"""
        seed       = 0
        no_train   = False
        fashion    = True                #weather to use fashion MNIST or normal MNIST
        batch_size = 128
        n_epochs   = 50
        lr         = 0.001
        n_steps    = 1000
        min_beta   = 10 ** -4
        max_beta   = 0.02              # Originally used by the authors
        n_samples  = 100               #number of samples to generate while testing
        time_emb_dim = 100

        """Unet params"""

        ks_b = 3
        ks_upd = 4
        
        s_b = 1
        s_upd = 2

        pad = 1 
        
        act = nn.SiLU()


    elif dataset == "turbRot":
        h,w = 64, 64

        """Directories"""

        data_dir    = "01_Data/TurbRot_data/omega8_1comp.h5"
        checkpt_dir = "02_checkpoints/"
        pred_dir    = "03_preds/"
        results_dir = "04_results/" 
        
        """Results_paths"""
        checkpt_path    =  "ddpm_model_turbRot.pt"
        video_path      =  "ddpm_turbRot.mp4"

        """Training and Testing parameters"""
        seed       = 0
        no_train   = False
        batch_size = 128
        n_epochs   = 50
        lr         = 0.001
        n_steps    = 1000
        min_beta   = 10 ** -4
        max_beta   = 0.02              # Originally used by the authors
        n_samples  = 100               #number of samples to generate while testing
        time_emb_dim = 100

        """Unet params"""

        ks_b = 3
        ks_upd = 4
        
        s_b = 1
        s_upd = 2

        pad = 1 
        
        act = nn.SiLU()



