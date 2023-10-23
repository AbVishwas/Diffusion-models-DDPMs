import torch
import torch.nn as nn
from utils.embedding import sinusoidal_embedding
from config import config


class MyBlock(nn.Module):
  def __init__(self, shape, in_c, out_c, kernel_size = config.ks_b, stride= config.s_b, padding= config.pad, activation = None, normalize = True):
      super(MyBlock, self).__init__()

      self.ln = nn.LayerNorm(shape)
      self.conv1 = nn.Conv2d(in_c,  out_c, kernel_size, stride, padding)
      self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding )
      self.activation = config.act if activation is None else activation
      self.normalize = normalize


  def forward(self, x):
      #print(f"shape of x:{x.shape}")
      out = self.ln(x) if self.normalize is True else x
      out = self.conv1(out)
      out = self.activation(out)
      out = self.conv2(out)
      out = self.activation(out)

      return out


"""
def convDim(x , k , p , s):
    x_new = (x -k + 2*p)/s + 1
    
    return x_new
"""

class MyUnet(nn.Module):
  def __init__(self, n_steps = config.n_steps, time_emb_dim = config.time_emb_dim):
      super(MyUnet, self).__init__()


      self.time_embed = nn.Embedding(n_steps, time_emb_dim)            #starting a instance of nn.embeding of req size
      self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)    #initialize the embedding layer with these positional embeddings
      self.time_embed.requires_grad_(False)
      

      h = config.h
      w = config.w
      
      ks_b   = config.ks_b
      ks_upd = config.ks_upd

      s_b   = config.s_b
      s_upd = config.s_upd

      pad = config.pad

      h1 = int(h/2)
      h2 = int(h/4)
      h3 = int(h/8)

      w1 = int(w/2)
      w2 = int(w/4)
      w3 = int(w/8)

      #down block
      self.te1 = self._make_te(time_emb_dim, 1)              #one embeding at each time step for all the images in dataset
      self.b1  = nn.Sequential(
          MyBlock((1,  h, w), 1,  10 ),                     #channel depth 1 > 10 > 10
          MyBlock((10, h, w), 10, 10 ),
          MyBlock((10, h, w), 10, 10 )
      )

      self.down1 = nn.Conv2d(10, 10, ks_upd, s_upd, pad)                # to downsample from 28 x 28 to 14 x 14

      self.te2 = self._make_te(time_emb_dim, 10)
      self.b2  = nn.Sequential(
          MyBlock((10, h1, w1), 10, 20),
          MyBlock((20, h1, w1), 20, 20),
          MyBlock((20, h1, w1), 20, 20)
      )

      self.down2 = nn.Conv2d(20, 20, ks_upd, s_upd, pad)


      self.te3 = self._make_te(time_emb_dim, 20)
      self.b3  = nn.Sequential(
          MyBlock((20, h2,  w2), 20, 40),
          MyBlock((40, h2,  w2), 40, 40),
          MyBlock((40, h2,  w2), 40, 40)
      )

      self.down3 = nn.Sequential(
         # nn.Conv2d(40, 40, 2, 1),
          nn.SiLU(),
          nn.Conv2d(40, 40, ks_upd, s_upd, pad)
        )


      # Bottleneck

      self.te_mid = self._make_te(time_emb_dim, 40)
      self.b_mid  = nn.Sequential(
          MyBlock((40, h3, w3), 40, 20),
          MyBlock((20, h3, w3), 20, 20),
          MyBlock((20, h3, w3), 20, 40)
      )

      # Up Block

      self.up1 = nn.Sequential(
          nn.ConvTranspose2d(40, 40, ks_upd, s_upd, pad),
          nn.SiLU()
          #nn.ConvTranspose2d(40, 40, 2, 1)
        )


      self.te4   = self._make_te(time_emb_dim, 80)
      self.b4 = nn.Sequential(
          MyBlock((80, h2, w2), 80, 40),
          MyBlock((40, h2, w2), 40, 20),
          MyBlock((20, h2, w2), 20, 20)
      )

      self.up2 = nn.ConvTranspose2d(20, 20, ks_upd, s_upd, pad)
      self.te5   = self._make_te(time_emb_dim, 40)
      self.b5  = nn.Sequential(
          MyBlock((40, h1, w1), 40, 20),
          MyBlock((20, h1, w1), 20, 10),
          MyBlock((10, h1, w1), 10, 10)
      )


      self.up3 = nn.ConvTranspose2d(10, 10, ks_upd, s_upd, pad)
      self.te_out = self._make_te(time_emb_dim, 20)
      self.b_out  = nn.Sequential(
          MyBlock((20, h, w), 20, 10),
          MyBlock((10, h, w), 10, 10),
          MyBlock((10, h, w), 10, 10, normalize = False)
      )

      self.conv_out = nn.Conv2d(10, 1, ks_b, s_b, pad)


  def forward(self, x, t):
      # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension (1 channel + 1 embed ))
      t =self.time_embed(t)  # 1 x emb_dim
      n = len(x)             # number of images

      """We are applying time embedding at each Unet stage starting from input image
        Hence shape of time embeding changes evrey time"""

      out1 = self.b1(        x        + self.te1(t).reshape(n, -1, 1, 1))            # (N, 10, 28, 28)
      out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))            # (N, 20, 14, 14)
      out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))            # (N, 40, 7, 7)

      out_mid =self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))    # (N, 40, 3, 3)

      out4 = torch.cat((out3, self.up1(out_mid)), dim=1)                             # (N, 80, 7, 7)
      out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))                        # (N, 20, 7, 7)

      out5 = torch.cat((out2, self.up2(out4)), dim=1)                                # (N, 40, 14, 14)
      out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))                        # (N, 10, 14, 14)

      out = torch.cat((out1, self.up3(out5)), dim=1)                                 # (N, 20, 28, 28)
      out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))                    # (N, 1, 28, 28)
      
      #print(f"shape of out just before last layer:{out.shape}")
      out = self.conv_out(out)
      #print(f"shape of out just after last layer:{out.shape}")
      return out



  def _make_te(self, dim_in, dim_out):                  #function (MLP) to map each time step to each embedding dim
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.SiLU(),
        nn.Linear(dim_out, dim_out))