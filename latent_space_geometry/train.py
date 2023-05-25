import os

import torch
from torchvision import datasets, models
from torchvision.transforms import ToTensor
from torchvision import transforms

import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm 

from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

import argparse
from einops import rearrange

from utils import CNN, CNN_cifar, Generator, Discriminator, get_next_batch, return_z, \
                Generator_resnet, Discriminator_resnet, cal_gradient_penalty


def train(gen, optimizer_g, disc, optimizer_d, train_loader, grad_penalty, dataiter, device,name,d_step,z_dim,parser):
    # Train the model
    for n in range(parser.n_steps):
        for d in range(d_step):
            real, labels, dataiter = get_next_batch(dataiter, train_loader)
            real = real.cuda(device)
            if parser.conditional:
                labels=labels.cuda(device)
                labels = torch.nn.functional.one_hot(labels, 10)
            else:
                labels=None
            fake = gen(return_z(real.shape[0],z_dim).cuda(device),labels)
            real_d, fake_d = disc(real,labels).mean(), disc(fake,labels).mean()
            emd = real_d - fake_d
            gradient_penalty, gradients = cal_gradient_penalty(disc,real,fake,device,labels=labels)
            loss = (- emd)
            if grad_penalty:
                loss+= gradient_penalty
            optimizer_d.zero_grad()      
            loss.backward()    
            optimizer_d.step() 
        
        z = return_z(real.shape[0],z_dim).cuda(device)
        fake = gen(z,labels)
        loss = - disc(fake,labels).mean()

        optimizer_g.zero_grad()     
        loss.backward()    
        optimizer_g.step()
        
        if n%20 == 0:
            print('Step : ',str(n))
            print('real_d : ',real_d)
            print('fake_d : ',fake_d)
            print('gradient_penalty : ',gradient_penalty,flush=True)
        
        if n%1000 == 0:
            figure = plt.figure(figsize=(5, 4))
            cols, rows = 10, 10
            if fake.shape[0] > 100:
                for i in range(1, cols * rows + 1):
                    img = (fake[i]).detach().cpu()
                    figure.add_subplot(rows, cols, i)
                    plt.axis("off")
                    if img.shape[0] == 3:
                        img = img/2+0.5
                        img = rearrange(img,'c h w -> h w c')
                        img = torch.clip(img,0,1)
                        plt.imshow(img.squeeze())
                    else:
                        plt.imshow(img.squeeze(), cmap="gray")
                plt.savefig(name+'/gen_' + str(n) + '.png')
                plt.close()
        
        if n%1000 == 0:
            path = name+'/gen_' + str(n) + '.pth'
            torch.save(gen.state_dict(), path)
            

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=1,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=0.00005,
    )
    parser.add_argument(
        "--lr_d",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--d_step",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100001,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--gen",
        type=str,
        default="cnn",
    )
    parser.add_argument(
        "--conditional",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="noise layer",
    )
    return parser
    


def main():
    parser = get_parser()
    parser, unknown = parser.parse_known_args()
    print(parser)
    name = parser.name
    depth_g = parser.depth

    d_step = parser.d_step
    lr_g = parser.lr_g
    lr_d = parser.lr_d
    batch_size = 256
    z_dim = parser.z_dim
    grad_penalty = True


    device = parser.device
    print('Creating folder : ',parser.name)
    os.makedirs(parser.name, exist_ok = True) 
    
    betas = (0.5,0.5)
    if parser.dataset == 'mnist':
        gen = Generator(z_dim=z_dim,nc=128,depth=depth_g).cuda(device)
        disc = Discriminator().cuda(device)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
        test_data = datasets.MNIST(
            root = 'data', 
            train = False, 
            transform = ToTensor()
        )
    elif parser.dataset == 'cifar10' or parser.dataset == 'synthetic_cifar10':
        if parser.conditional:
            nclasses = 10
        else:
            nclasses = 0
        batch_size=256
        betas = (0.,0.999)
        gen = Generator_resnet(z_dim=z_dim,nc=parser.width,nclasses=nclasses).cuda(device)
        disc = Discriminator_resnet(nclasses=nclasses).cuda(device)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        if parser.dataset == 'synthetic_cifar10':
            train_data = datasets.ImageFolder(root='data/synthetic_cifar10/', transform=transform)
            test_data = datasets.ImageFolder(root='data/synthetic_cifar10/', transform=transform)
        else:
            train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    elif parser.dataset == 'cifar100':
        batch_size=256
        betas = (0.,0.999)
        gen = Generator_resnet(z_dim=z_dim,nc=128).cuda(device)
        disc = Discriminator_resnet().cuda(device)
        optimizer_g = optim.Adam(gen.parameters(), lr = lr_g, betas = betas)
        optimizer_d = optim.Adam(disc.parameters(), lr = lr_d, betas = betas)
        
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        train_data = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)
    print(train_data)
    print(test_data)

    loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=15),

    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    }

    train_loader = loaders['train']
    dataiter = iter(train_loader)
    print(gen)
    print(disc)
    train(gen, optimizer_g, disc, optimizer_d, train_loader, grad_penalty, dataiter, device, name, d_step,z_dim,parser)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()