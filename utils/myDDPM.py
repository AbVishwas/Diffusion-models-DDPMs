import torch
import torch.nn as nn
from tqdm.auto import tqdm
from utils.post_proc_images import show_images, generate_new_images

class DDPM(nn.Module):
    def __init__(self, network, n_steps= 200, min_beta = 10**-4, max_beta= 0.02, device=None, image_chw = (1,64,64)):
          super(DDPM, self).__init__()
          self.n_steps = n_steps
          self.device = device
          self.image_chw = image_chw
          self.network = network.to(device)
          self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)   #schedular for noise addition
          self.alphas = 1 - self.betas
          self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1 ]) for i in range(len(self.alphas))]).to(device)   # list of product of (1-betas) for each time step


    def forward(self, x0, t , eta =None):
          """Noisy image in one step"""

          n, c , h, w = x0.shape
          a_bar =self.alpha_bars[t]

          if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)
          #formula
          noisy = a_bar.sqrt().reshape(n,1,1,1)*x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1)*eta

          return noisy

    def backward(self, x, t):
        #Run each image through the network for each timestep t in the vector t
        #The network returns its estimation of the noise that was added

        return self.network(x, t)





def training_loop(ddpm, loader, n_epochs, optim, device, display= False, store_path=""):
    mse = nn.MSELoss()
    best_loss = float("inf")           #represents positive infinity, which is a special floating-point value that represents a quantity that is larger than any finite number
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc =f"Training proecess", colour="#00ff00"):
        epoch_loss =0.0
        for step, batch in enumerate(loader):  #enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch +1}/{n_epochs}", colour="#005500")):
            #load data
                       
            x0 = batch.to(device).float()
            
            n = len(x0)

            #print(f"shape of batch (x0): {x0.shape} and {x0.dtype}, image: {x0[0].dtype}")

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            #print(f"eta:{eta.shape}")
            t = torch.randint(0, n_steps, (n,)).to(device)

            #Computing the noisy image based on x0 and teh teimestep (forward step)
            noisy_imgs = ddpm.forward(x0, t, eta)
            #print(f"noisy image:{noisy_imgs.shape}")

            #Getting models estimation of noise based on noisy images and time step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))
            #print(f"eta theta:{eta_theta.shape}")

            #optimizer

            
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

          # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device)[0], f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)