import torch 

class config:
    seed = 0

    data_dir    = "01_Data/"
    checkpt_dir = "02_checkpoints/"
    pred_dir    = "03_preds/"
    results_dir = "04_results/" 
    

    checkpt_path    =  "ddpm_model_fashion.pt"
    video_path      = "ddpm_fashion.mp4"

    #params
    no_train   = False
    fashion    = True                #weather to use fashion MNIST or normal MNIST
    batch_size = 128
    n_epochs   = 50
    lr         = 0.001
    n_steps    = 1000
    min_beta   = 10 ** -4
    max_beta   = 0.02              # Originally used by the authors
    n_samples  = 100               #number of samples to generate while testing
    
    #device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")