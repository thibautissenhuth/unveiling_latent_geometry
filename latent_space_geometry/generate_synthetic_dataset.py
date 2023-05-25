import os
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import numpy as np
from torch.nn.utils import spectral_norm 

from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

import argparse
from einops import rearrange

from utils import CNN, CNN_cifar, Generator, get_next_batch, return_z, Generator_resnet

from torch import optim
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

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
        "--model_path",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=50000,
        help="seed for seed_everything",
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
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--conditional",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="noise layer",
    )
    parser.add_argument(
        "--gen",
        type=str,
        default="cnn",
    )
    return parser

def return_labels(batch_size):
    y = torch.arange(10)
    y = y.unsqueeze(1).repeat(1,batch_size//10).reshape(-1)
    y = torch.nn.functional.one_hot(y, 10)
    return y


def save_batch_img(img_batch,index,folder):
    batch_size = img_batch.shape[0]
    img_batch = rearrange(img_batch,'b c h w -> b h w c')
    img_batch = (img_batch+1)/2
    for i in range(batch_size):
        class_i = int(i/10)
        idx_i = batch_size * index + i 
        img = img_batch[i].cpu().numpy()
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img)
        print(folder+'/'+str(class_i)+'/'+str(idx_i)+'.jpeg')
        img.save(folder+'/'+str(class_i)+'/'+str(idx_i)+'.jpeg')

def main():
    parser = get_parser()
    parser, unknown = parser.parse_known_args()
    print(parser)
    dataset = parser.dataset
    z_dim = parser.z_dim
    gen = parser.gen
    device = parser.device
    if dataset == 'cifar10':
        if parser.conditional:
            nclasses = 10
        gen = Generator_resnet(z_dim=z_dim,nc=128,nclasses=nclasses).cuda(parser.device)
    else:
        gen = Generator(z_dim=z_dim,nc=128).cuda(parser.device)
                   
    gen.load_state_dict(torch.load(parser.model_path,map_location='cuda:'+str(parser.device)))
    gen.eval()
    
    os.makedirs(parser.target_folder, exist_ok = True)
    for i in range(10):
        os.makedirs(parser.target_folder+'/'+str(i), exist_ok = True)
    
    print('Saving dataset in : ',parser.target_folder)
    batch_size = 100
    n_iterations = parser.n_points // batch_size
    
    with torch.no_grad():
        for i in range(n_iterations):
            z = return_z(batch_size, z_dim).cuda(parser.device)
            y = return_labels(batch_size).cuda(parser.device) 
            gen_img = gen(z,y)
            save_batch_img(gen_img,i,parser.target_folder)
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()