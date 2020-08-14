"""model.py"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

    
class Encoder_H(nn.Module):
    def __init__(self, input_shape=(64,64), z_dim=10, nc=3, padding=1):
        super(Encoder_H, self).__init__()
        self.conv2d_1 = nn.Conv2d(nc, 32, 4, 2, padding)
        self.conv2d_2 = nn.Conv2d(32, 32, 4, 2, padding)
        self.conv2d_3 = nn.Conv2d(32, 64, 4, 2, padding)
        self.conv2d_4 = nn.Conv2d(64, 64, 4, 2, padding)
            
        self.flatten_shape, self.dconv_size = self._get_conv_output(input_shape, nc)
        self.linear = nn.Linear(self.flatten_shape, z_dim*2)
        
    # generate input sample and forward to get shape
    def _get_conv_output(self, shape, nc):
        bs = 1
        dummy_x = torch.empty(bs, nc, *shape)
        x, dconv_size = self._forward_features(dummy_x)
        flatten_shape = x.flatten(1).size(1)
        return flatten_shape, dconv_size
    
    def _forward_features(self, x):
        size0 = x.shape[1:]
        x = torch.relu(self.conv2d_1(x))
        size1 = x.shape[1:]
        x = torch.relu(self.conv2d_2(x))
        size2 = x.shape[1:]
        x = torch.relu(self.conv2d_3(x))
        size3 = x.shape[1:]
        x = torch.relu(self.conv2d_4(x))  
        size4 = x.shape[1:]
        return x, [size0,size1,size2,size3,size4]
    
    def forward(self, x):
        x = torch.relu(self.conv2d_1(x))
        x = torch.relu(self.conv2d_2(x))
        x = torch.relu(self.conv2d_3(x))
        x = torch.relu(self.conv2d_4(x))
        x = self.linear(x.flatten(1))
        return x
        
        
class Decoder_H(nn.Module):
    def __init__(self, output_shape, z_dim=10, nc=3, padding=1):
        super(Decoder_H, self).__init__()
        self.output_shape = output_shape
        flatten_shape = output_shape[-1][0]*output_shape[-1][1]*output_shape[-1][2]
        self.linear = nn.Linear(z_dim, flatten_shape)
        self.conv2d_1 = nn.ConvTranspose2d(64, 64, 4, 2, padding)
        self.conv2d_2 = nn.ConvTranspose2d(64, 32, 4, 2, padding)
        self.conv2d_3 = nn.ConvTranspose2d(32, 32, 4, 2, padding)
        self.conv2d_4 = nn.ConvTranspose2d(32, nc, 4, 2, padding)
 
    def _forward_features(self, x):
        x = torch.relu(self.conv2d_1(x, self.output_shape[3][1:]))
        x = torch.relu(self.conv2d_2(x, self.output_shape[2][1:]))
        x = torch.relu(self.conv2d_3(x, self.output_shape[1][1:]))
        x = self.conv2d_4(x, self.output_shape[0][1:])
        return x
    
    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = x.view(-1, *self.output_shape[4])
        x = self._forward_features(x)
        return x

class BetaVAE_H(nn.Module):
    def __init__(self, input_shape=(64,64), z_dim=10, nc=3, padding=0, activation=nn.Identity()):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.activation = activation
        
        self.encoder = Encoder_H(input_shape=input_shape, nc=nc, z_dim=z_dim, padding=padding)
        self.decoder = Decoder_H(self.encoder.dconv_size, nc=nc, z_dim=z_dim, padding=padding)
        
    def forward(self,x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z)
        
        return self.activation(x_recon), mu, logvar
    
    
class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
