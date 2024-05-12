''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
  
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        Process and downscale the image feature maps
        '''
        # Only applying a single convolution and changing pooling stride to 1 to reduce dimension reduction
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2, stride=2)  # Changed stride to 1
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),  # Adjusted in_channels here
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )
    def forward(self, x, skip):
        # Adjust the size of cemb1 to match the size of x
        cemb1_resized = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, cemb1_resized), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class SigmaEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(SigmaEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, emb_dim),  # Adjust input_dim to match flattened input
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        # Ensure x is flattened correctly to match the expected input dimensions
        x = x.view(x.size(0), -1)
        return self.fc(x)


    
class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(2), nn.GELU())

        self.timeembed1 = EmbedFC(1, n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.sigma_embedding1 = SigmaEmbedding(144, n_feat)  
        self.sigma_embedding2 = SigmaEmbedding(144, 256) 

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(n_feat, 2 * n_feat, 7, 7),  
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(3 * self.n_feat, self.n_feat)  # Adjust the in_channels here
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x, sigma_is, t):
        x = self.init_conv(x)
        print("x shape: ", x.shape)
        down1 = self.down1(x)
        print("down1 shape: ", down1.shape)
        down2 = self.down2(down1)
        print("down2 shape: ", down2.shape)
        hiddenvec = self.to_vec(down2)
        print("hiddenvec shape: ", hiddenvec.shape)
        # # Embedding time and sigma_is
        # temb1 = self.timeembed1(t).view(-1, self.n_feat, 1, 1)
        # semb1 = self.sigma_embedding1(sigma_is).view(-1, self.n_feat, 1, 1)
        # # temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        # # semb2 = self.sigma_embedding2(sigma_is).view(-1, self.n_feat, 1, 1)
        # print("temb shape: ", temb1.shape)
        # print("semb shape: ", semb1.shape)

        # hiddenvec = torch.cat((hiddenvec, temb1, semb1), 1)  # Concatenate embeddings

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        # up2 = self.up1(semb1*up1+ temb1, down2, 2)  # add and multiply embeddings
        # up3 = self.up2(up2, down1)
        up3_resized = F.interpolate(up2, size=x.size()[2:], mode='bilinear', align_corners=True)

        out = self.out(torch.cat((up3_resized, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, sigma_is):
        """
        this method is used in training, so samples t and noise randomly
        """
        print(x.shape)
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x +
            self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps

        # Dropout sigma with some probability
        sigma_mask = torch.bernoulli(torch.full_like(sigma_is, 1 - self.drop_prob)).to(self.device)
        masked_sigma = sigma_is * sigma_mask
        print("sigma_ic shape: ",masked_sigma.shape)
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, masked_sigma, _ts / self.n_T))

    def sample(self, n_sample, size, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        sigma_i = torch.randn(n_sample, 1).to(device)  # Sample some initial sigma values

        # Don't drop sigma at test time
        sigma_mask = torch.zeros_like(sigma_i).to(device)

        # Double the batch
        sigma_i = sigma_i.repeat(2, 1)
        sigma_mask = sigma_mask.repeat(2, 1)
        sigma_mask[n_sample:] = 1.  # Makes second half of batch sigma free

        x_i_store = []  # Keep track of generated steps
        for i in range(self.n_T, 0, -1):
            t_is = torch.full((n_sample * 2, 1, 1, 1), i / self.n_T, device=device)

            # Repeat noise and timestep for double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # Split predictions and compute weighting
            eps = self.nn_model(x_i, sigma_i, t_is)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store




# def train_mnist():

#     # hardcoding these here
#     n_epoch = 1
#     batch_size = 64
#     n_T = 400 # 500
#     device = "cuda:0"
#     n_classes = 10
#     n_feat = 128 # 128 ok, 256 better (but slower)
#     lrate = 1e-4
#     save_model = False
#     save_dir = './data/diffusion_outputs10/'
#     ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

#     ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
#     ddpm.to(device)

#     # optionally load a model
#     # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

#     tf = transforms.Compose([
#     transforms.Resize((12, 12)),  # Resize to 12x12
#     transforms.Grayscale(num_output_channels=3),  # Convert grayscale to "RGB" by replicating channels
#     transforms.ToTensor()  # Convert image data to a tensor
#     ])

#     dataset = MNIST("./data", train=True, download=True, transform=tf)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
#     optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

#     for ep in range(n_epoch):
#         print(f'epoch {ep}')
#         ddpm.train()

#         # linear lrate decay
#         optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

#         pbar = tqdm(dataloader)
#         loss_ema = None
#         for x, c in pbar:
            
#             optim.zero_grad()
#             x = x.to(device)
#             c = c.to(device)
#             print(x.shape)
#             print(c.shape)
#             loss = ddpm(x, c)
#             loss.backward()
#             if loss_ema is None:
#                 loss_ema = loss.item()
#             else:
#                 loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
#             pbar.set_description(f"loss: {loss_ema:.4f}")
#             optim.step()
        
#         # for eval, save an image of currently generated samples (top rows)
#         # followed by real images (bottom rows)
#         ddpm.eval()
#         with torch.no_grad():
#             n_sample = 4*n_classes
#             for w_i, w in enumerate(ws_test):
#                 x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

#                 # append some real images at bottom, order by class also
#                 x_real = torch.Tensor(x_gen.shape).to(device)
#                 for k in range(n_classes):
#                     for j in range(int(n_sample/n_classes)):
#                         try: 
#                             idx = torch.squeeze((c == k).nonzero())[j]
#                         except:
#                             idx = 0
#                         x_real[k+(j*n_classes)] = x[idx]

#                 x_all = torch.cat([x_gen, x_real])
#                 grid = make_grid(x_all*-1 + 1, nrow=10)
#                 # save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
#                 # print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

#                 if ep%5==0 or ep == int(n_epoch-1):
#                     # create gif of images evolving over time, based on x_gen_store
#                     fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
#                     def animate_diff(i, x_gen_store):
#                         print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
#                         plots = []
#                         for row in range(int(n_sample/n_classes)):
#                             for col in range(n_classes):
#                                 axs[row, col].clear()
#                                 axs[row, col].set_xticks([])
#                                 axs[row, col].set_yticks([])
#                                 # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
#                                 plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
#                         return plots
#                     ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
#                     # ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
#                     # print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
#         # # optionally save model
#         # if save_model and ep == int(n_epoch-1):
#         #     torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
#         #     print('saved model at ' + save_dir + f"model_{ep}.pth")

# if __name__ == "__main__":
#     train_mnist()

