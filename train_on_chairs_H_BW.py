from model import BetaVAE_H
from torchvision.transforms import ToTensor, Resize, Compose, Grayscale
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid, save_image
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
from dataset import Chairs

from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver

ex = Experiment('Chair-VAE_H-z_10')

@ex.config
def config():
    lr = 5e-4
    epoches = 100
    beta = 4
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    image_size = 64
    z_dim=10
    recon_type = 'MSE'
    activation = 'sigmoid'
    warm_up = 0
    logdir = f'runs/Chair-BW-VAEv2_H-z_{z_dim}-{recon_type}-warm_up={warm_up}-{activation}-beat={beta}-'+ datetime.now().strftime('%y%m%d-%H%M%S')
    ex.observers.append(FileStorageObserver.create(logdir)) # saving source code
    
    
@ex.automain
def train(logdir, lr, epoches, beta, num_workers, image_size, z_dim, recon_type, activation, warm_up, device):   
    writer = SummaryWriter(logdir) # create tensorboard logger    
    
    if activation=='sigmoid':
        act_fn = nn.Sigmoid()
    elif activation == 'identity':
        act_fn = nn.Identity()
    else:
        print(f'Please choose a valid activation, {activation} is not defined\n'
              f"Using the default setting 'Identity'")
        
    
    def get_fig(batch):
        fig, axes = plt.subplots(4,8, figsize=(16,8))
        for i,ax in zip(batch.cpu().detach().numpy(),axes.flatten()):
            ax.imshow(i[0], cmap='binary_r')
            ax.axis('off')
        fig.tight_layout(pad=0.1)   
        return fig

    def reconstruction_loss(x, x_recon, recon_type):
        batch_size = x.size(0)
        assert batch_size != 0

        if recon_type == 'BCE':
            recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        elif recon_type == 'MSE':
            recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        else:
            recon_loss = None

        return recon_loss

    def kl_divergence(mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = torch.sum(klds)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld
        
    print('loading dataset...', end='\r')
    
    dataset = Chairs('./data', transform=Compose([Resize((image_size, image_size)),Grayscale(),ToTensor()]))
    torch.manual_seed(0)
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-5000,5000]) 
    torch.manual_seed(torch.initial_seed())
    train_loader = DataLoader(train_set,batch_size=32,shuffle=True,num_workers=num_workers,
                            pin_memory=False,drop_last=True)
    test_loader = DataLoader(test_set,batch_size=32,shuffle=False,num_workers=num_workers,
                            pin_memory=False,drop_last=True)

    x = next(iter(dataset))
    net = BetaVAE_H(input_shape=x.shape[1:],nc=x.size(0), z_dim=10, padding=0, activation=act_fn)
    net = net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    
    loss_dict = {'train_recon':[], 'train_kld':[], 'test_recon':[], 'test_kld':[]}

    for e in range(1,epoches+1):
        for key in loss_dict: # Resetting dict
            loss_dict[key]=[]

        for idx, (x) in enumerate(train_loader):      
            x = x.to(device)
            x_recon, mu, logvar = net(x)
            
            if e-1 < warm_up:
                recon_loss = reconstruction_loss(x, x_recon, recon_type)
                total_kld = torch.tensor([0])
                beta_vae_loss = recon_loss                
            else:
                recon_loss = reconstruction_loss(x, x_recon, recon_type)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)    
                beta_vae_loss = recon_loss + beta*total_kld

            optim.zero_grad()
            beta_vae_loss.backward()
            optim.step()
            
            loss_dict['train_recon'].append(recon_loss.item())
            loss_dict['train_kld'].append(total_kld.item())

            print(f'Training ep [{e}/{epoches}], [{idx}/{len(train_loader)}] '
                  f'recon_loss = {recon_loss.item():.3f}, total_kld = {total_kld.item():.3f}', end='\r')            

        for idx, (x) in enumerate(test_loader):   
            x = x.to(device)
            x_recon, mu, logvar = net(x)
            recon_loss = reconstruction_loss(x, x_recon, recon_type)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)    

            loss_dict['test_recon'].append(recon_loss.item())
            loss_dict['test_kld'].append(total_kld.item())

            print(f'Testing ep [{e}/{epoches}], [{idx}/{len(test_loader)}] '
                  f'recon_loss = {recon_loss.item():.3f}, total_kld = {total_kld.item():.3f}', end='\r')

        if e==1: # Showing the original transcription and spectrograms
            fig_x = get_fig(x)
            writer.add_figure('images/Original', fig_x , e)
        if e%2==0:
            fig_recon = get_fig(x_recon)
            writer.add_figure('images/Reconstruction', fig_recon , e)
            writer.add_scalar(f'test/mu', mu[0,0], global_step=e)
            writer.add_scalar(f'test/logvar', logvar[0,0], global_step=e)
            for key in loss_dict:
                name = key.split('_')
                writer.add_scalar(f'{name[0]}/{name[1]}', np.mean(loss_dict[key]), global_step=e)
        if e%10==0:
            torch.save(net.state_dict(), os.path.join(logdir, f'model-{e}.pt'))    
            torch.save(optim.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))