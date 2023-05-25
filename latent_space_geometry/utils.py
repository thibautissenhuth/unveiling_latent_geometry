import torch

import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

import argparse
from einops import rearrange

class Generator_resnet(nn.Module):
    def __init__(self, nc = 128, z_dim=128, nclasses=0):
        super(Generator_resnet, self).__init__()
        self.nc = nc
        self.nclasses = nclasses
        if self.nclasses>0:
            z_dim = z_dim + nclasses
        self.dense = nn.Sequential(nn.Linear(z_dim, 4*4*nc))

        self.resblock_1 = ResBlock_generator(nc,nc, act_function=nn.ReLU(), norm=True, z_dim=z_dim)
        self.resblock_2 = ResBlock_generator(nc,nc, act_function=nn.ReLU(), norm=True, z_dim=z_dim)
        self.resblock_3 = ResBlock_generator(nc,nc, act_function=nn.ReLU(), norm=True, z_dim=z_dim)
        
        output_nc = 3
        self.conv_final = nn.Sequential(
            nn.Conv2d(nc, output_nc, kernel_size=3, padding=1),
            nn.Tanh())
                
        self.interpol_1 = Interpolate(size = (8,8), mode = 'nearest')
        self.interpol_2 = Interpolate(size = (16,16), mode = 'nearest')
        self.interpol_3 = Interpolate(size = (32,32), mode = 'nearest')
        

    def forward(self, z, labels=None):
        if labels is not None:
            z = torch.cat((z,labels),1)
        img = self.dense(z)
        img = img.reshape(z.shape[0],self.nc,4,4)
        img = self.interpol_1(self.resblock_1(img,z))
        img = self.interpol_2(self.resblock_2(img,z))
        img = self.interpol_3(self.resblock_3(img,z))
        img = self.conv_final(img)
        return img

class Discriminator_resnet(nn.Module):
    def __init__(self, config=None, nclasses=0):
        super(Discriminator_resnet, self).__init__()
        input_nc = 3
        self.config = config
        nc = 256
        self.resblock_1 = ResBlock(3,nc, act_function=nn.ReLU())
        self.resblock_2 = ResBlock(nc,nc, act_function=nn.ReLU())
        self.resblock_3 = ResBlock(nc,nc, act_function=nn.ReLU())
        self.resblock_4 = ResBlock(nc,nc, act_function=nn.ReLU())
        self.down_pool = nn.AvgPool2d((2,2))
        self.nclasses = nclasses

        if self.nclasses > 0:
            self.linear = nn.Sequential(nn.Linear(nc+nclasses,nc),
                                       nn.ReLU(),nn.Linear(nc,1,bias=False))
        else:
            self.linear = nn.Sequential(nn.Linear(nc,1))

    def forward(self, x, labels=None):
        x = self.down_pool(self.resblock_1(x))
        x = self.down_pool(self.resblock_2(x))
        x = self.resblock_4(self.resblock_3(x))
        x = (torch.sum(torch.sum(x, dim = 3), dim = 2))/(8*8) ##Mean pooling
        if labels is not None:
            x = torch.cat((x,labels),1)
        #x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class ResBlock_generator(nn.Module):
    def __init__(self, in_planes, planes, act_function=nn.ReLU(), norm=True, stride=1, config = None, z_dim=128):
        super(ResBlock_generator, self).__init__()
        self.act_function = act_function
        self.norm = norm
        self.in_planes = in_planes
        self.planes = planes
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        if self.norm:
            self.bn1 = adaptiveBatchNorm2D(planes,z_dim=z_dim)
            self.bn2 = adaptiveBatchNorm2D(planes,z_dim=z_dim)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()

        if in_planes != planes:
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0))
            if self.norm:
                self.bn3 = nn.adaptiveBatchNorm2D(planes,z_dim=z_dim)

    def forward(self, x, z):
        out = self.bn2(self.conv2( self.act_function(self.bn1(self.conv1(x),z))), z)
        if self.in_planes != self.planes:
            out = out + self.bn3(self.conv3(x), z)
        else:
            out = out + x
        out = self.act_function(out)
        return out

class adaptiveBatchNorm2D(nn.Module):
    def __init__(self,planes,z_dim=128):
        super(adaptiveBatchNorm2D,self).__init__()
        self.bn = nn.BatchNorm2d(planes,affine = False)
        self.gamma = nn.Sequential(nn.Linear(z_dim,planes),
        nn.ReLU(),nn.Linear(planes,planes,bias=False))
        self.beta = nn.Sequential(nn.Linear(z_dim,planes),
        nn.ReLU(),nn.Linear(planes,planes,bias=False))

    def forward(self, x, z):
        gamma_z = self.gamma(z).unsqueeze(2).unsqueeze(3)
        beta_z = self.beta(z).unsqueeze(2).unsqueeze(3)
        return gamma_z * self.bn(x) + beta_z

class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, act_function=nn.LeakyReLU(0.2), stride=1):
        super(ResBlock, self).__init__()
        self.act_function = act_function
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1),
                self.act_function)
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))

        if in_planes == planes:
            self.conv3 = nn.Sequential()
        else:
            self.conv3 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = out + self.conv3(x)
        out = self.act_function(out)
        return out

class CNN_cifar(nn.Module):
    def __init__(self, input_nc = 1, hidden_nc = 128, nclasses=10):
        super(CNN_cifar, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=input_nc,              
                out_channels=hidden_nc,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.GELU(),
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),     
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),     
            nn.GELU(),
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),     
            nn.GELU(),
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),     
            nn.GELU(),                      
        )
        
        self.linear = nn.Sequential(nn.Linear(hidden_nc * 8 * 8, 100),
                                     nn.GELU())
        self.out = nn.Linear(100,nclasses)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)  
        x = self.linear(x)
        output = self.out(x)
        return output, x    # return x for visualization

class CNN(nn.Module):
    def __init__(self, input_nc = 1, hidden_nc = 128):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=input_nc,              
                out_channels=hidden_nc,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(hidden_nc, hidden_nc, 3, 1, 1),     
            nn.ReLU(),                      
        )
        if input_nc == 1:
            self.linear = nn.Sequential(nn.Linear(hidden_nc * 7 * 7, 100),
                                     nn.ReLU())
        else:
            self.linear = nn.Sequential(nn.Linear(hidden_nc * 8 * 8, 100),
                                     nn.ReLU())
        self.out = nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)  
        x = self.linear(x)
        output = self.out(x)
        return output, x    # return x for visualization

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        input_nc = 1
        hidden_nc = 512
        self.conv = nn.Sequential(
                nn.Conv2d(input_nc, hidden_nc, kernel_size=4, stride = 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=4, stride = 2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            
        self.linear = nn.Sequential(nn.Linear(hidden_nc*7*7,1,bias=False))

    def forward(self, x, labels=None):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
    
    
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

class Mapping(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, depth = 1):
        super(Mapping, self).__init__()
        self.nc = nc
        layers = []
        for d in range(depth-1):
            layers.append(nn.Linear(z_dim,z_dim))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
        layers.append(nn.Linear(z_dim, 7*7*nc))
        self.dense = nn.Sequential(*layers)
       
    def forward(self, z):
        z = self.dense(z)
        return z

class Synthesis(nn.Module):
    def __init__(self,z_dim = 128, nc = 128):
        super(Synthesis, self).__init__()
        self.nc = nc
        nc = nc//2
        self.conv_1 = nn.Sequential(
            nn.Conv2d(nc*2, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )


        nc = nc//2
        self.conv_2 = nn.Sequential(
                nn.Conv2d(nc*2, nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(nc, nc, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(nc, nc, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc, 1, kernel_size=3, padding=1))

        self.interpol_1 = Interpolate(size = (14,14), mode = 'nearest')
        self.interpol_2 = Interpolate(size = (28,28), mode = 'nearest')
        self.final_activation = nn.Tanh()

    def forward(self, z):
        img = z.reshape(z.shape[0],self.nc,7,7)
        img = self.conv_1(img)
        img = self.interpol_1(img)
        img = self.conv_2(img)
        img = self.interpol_2(img)
        img = self.conv_final(img)
        img = self.final_activation(img) / 2 + 0.5
        return img

class Generator(nn.Module):
    def __init__(self,z_dim = 128, nc = 128, depth = 1, tanh = 1):
        super(Generator, self).__init__()
        self.nc = nc
        self.mapping = Mapping(z_dim,nc,depth)
        self.synthesis = Synthesis(z_dim,nc)

    def forward(self, z, labels=None):
        z = self.mapping(z)
        img = self.synthesis(z)
        return img
    
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0,
                         config = None, labels=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand_like(real_data)
            #alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        if labels is None:
            disc_interpolates = netD(interpolatesv)
        else:
            disc_interpolates = netD(interpolatesv,labels)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
    
def get_next_batch(dataiter, train_loader):
    try:
        images, labels = dataiter.next()
    except StopIteration:
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
    return images, labels, dataiter

def return_z(batch_size, z_dim):
    z = torch.randn((batch_size,z_dim))
    return z