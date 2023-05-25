import os
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

import argparse
from einops import rearrange

from utils import CNN, CNN_cifar, Generator, Discriminator, get_next_batch, return_z, Generator_resnet

from torch import optim
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.stats
from scipy import linalg
from prdc import compute_prdc

@torch.no_grad()
def return_real_features(classifier,n_iterations,dataiter,train_loader,device):
    X = torch.tensor([])
    for i in range(n_iterations):
        img, _, _ = get_next_batch(dataiter, train_loader)
        img = img.cuda(device)
        with torch.no_grad():
            _,feats = classifier(img)
        X = torch.cat((X,feats.cpu()))
    return X
    
@torch.no_grad()
def return_gen_features(gen,classifier,batch_size,n_iterations,z_dim,device):
    X = torch.tensor([])
    for i in range(n_iterations):
        z = return_z(batch_size, z_dim).cuda(device)
        with torch.no_grad():
            gen_img = gen(z)
            _,feats = classifier(gen_img)
        X = torch.cat((X,feats.cpu()))
    return X

def get_mu_sigma(feature_array):
    return np.mean(feature_array,axis=0), np.cov(feature_array,rowvar=False)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Taken from: https://github.com/mseitzer/pytorch-fid
    
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def print_format_metric(metric):
    metric_m, metric_s = mean_confidence_interval(metric)
    print('%.1f $\\pm$ %.1f ' % (metric_m, metric_s))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def return_x_y(gen,classifier,batch_size,n_iterations,z_dim,device):
    X, Y = [], []
    for i in range(n_iterations):
        z = return_z(batch_size, z_dim).cuda(device)
        with torch.no_grad():
            gen_img = gen(z)
            labels,_ = classifier(gen_img)
            labels = torch.max(labels, 1)[1].data.squeeze()
        X.append(z.detach().cpu().numpy())    
        Y.append(labels.detach().cpu().numpy())
    X,Y = np.concatenate(X,axis=0), np.concatenate(Y,axis=0)
    return X, Y

def return_x_y_filtered(gen,classifier,batch_size,n_points,z_dim,device):
    X, Y = [], []
    count = 0
    while True:
        z = return_z(batch_size, z_dim).cuda(device)
        with torch.no_grad():
            gen_img = gen(z)
            labels,_ = classifier(gen_img)
            
            max_score = torch.nn.functional.softmax(labels,dim=1)
            max_score,_ = torch.max(max_score,dim=1)
            zeros = torch.zeros_like(max_score)
            idx_good_gen = torch.nonzero(torch.where(max_score > 0.8, max_score, zeros))
            
            z = z[idx_good_gen].squeeze(1)
            gen_img = gen_img[idx_good_gen].squeeze(1)
            labels = labels[idx_good_gen].squeeze(1)
            count += z.shape[0]
            
            labels = torch.max(labels, 1)[1].data.squeeze()
            
        X.append(z.detach().cpu().numpy())    
        Y.append(labels.detach().cpu().numpy())
        if count > n_points:
            X = X[:n_points]
            Y = Y[:n_points]
            break
    X,Y = np.concatenate(X,axis=0), np.concatenate(Y,axis=0)
    return X, Y

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # st()
    _, pred = output.topk(maxk, 1, True, True)
    #print(pred)
    pred = pred.t()
    # st()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct = (pred == target.view(1, -1).expand_as(pred))
    correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size * 100))
    return res

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
        "--folder_path",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
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
        "--gen",
        type=str,
        default="cnn",
    )
    parser.add_argument(
        "--logistic_reg",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="Logistic regression test",
    )
    parser.add_argument(
        "--convex_test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="Logistic regression test",
    )
    parser.add_argument('--gen_list', default=[], nargs='+')
    return parser

def main():
    parser = get_parser()
    parser, unknown = parser.parse_known_args()
    print(parser)
    dataset = parser.dataset
    z_dim = parser.z_dim
    gen = parser.gen
    device = parser.device
    folder_path = parser.folder_path
    gen_list = parser.gen_list
    gen_list = [os.path.join(folder_path,gen_path) for gen_path in gen_list]
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'synthetic_cifar10' or dataset == 'cifar3':
        gen = Generator_resnet(z_dim=z_dim,nc=128).cuda(parser.device)
        if dataset == 'cifar10' or dataset == 'synthetic_cifar10' or dataset == 'cifar3':
            classifier = CNN_cifar(input_nc = 3, hidden_nc = 256).cuda(parser.device)
            name_classif = 'models/cifar10_classifier.pth'
        else:
            classifier = CNN_cifar(input_nc = 3, hidden_nc = 256, nclasses=100).cuda(parser.device)
            name_classif = 'cifar100_classifier.pth'
        print('Classifer used : ',name_classif)
        classifier.load_state_dict(torch.load(name_classif))
    else:
        gen = Generator(z_dim=z_dim,nc=128).cuda(parser.device)
        classifier = CNN().cuda(0)
        classifier.load_state_dict(torch.load('models/mnist_classifier.pth'))
    classifier.eval()
    
    if parser.convex_test:
        print('Convexity test')
        if parser.dataset == 'cifar100':
            classes = np.arange(100)
            n_iter_per_class = 10
        else:  
            classes = np.arange(10)
            n_iter_per_class = 100
        accuracies = []
        for path_gen in gen_list:
            gen.load_state_dict(torch.load(path_gen,map_location='cuda:'+str(parser.device)))
            gen.eval()
            with torch.no_grad():
                labels, gt = torch.tensor([]), torch.tensor([])
                for y_i in classes:
                    for j in range(n_iter_per_class):
                        if 'cifar' in dataset:
                            X, Y = return_x_y_filtered(gen,classifier,200,200,z_dim,parser.device)
                        else:
                            X, Y = return_x_y(gen,classifier,200,1,z_dim,parser.device)
                        idx_yi = np.where(Y == y_i, 1, 0).nonzero()[0]
                        if len(idx_yi)<2:
                            continue
                        z_1 = torch.from_numpy(X[idx_yi[0]]).unsqueeze(0).repeat(3,1).cuda(device)
                        z_2 = torch.from_numpy(X[idx_yi[1]]).unsqueeze(0).repeat(3,1).cuda(device)
                        interpol_eps = torch.arange(start=0.25, end=0.76, step=0.25).unsqueeze(1).cuda(device)
                        z = z_1 * interpol_eps + (1-interpol_eps) * z_2

                        gen_img = gen(z)
                        labels_j,_ = classifier(gen_img)

                        labels = torch.cat((labels,labels_j.cpu()),0)
                        gt = torch.cat((gt,torch.zeros((3))+y_i),0)
                acc = accuracy(labels,gt)
                accuracies.append(acc)
    
    print_format_metric(accuracies)
    
    if parser.logistic_reg:
        print('Logistic regression in the latent space')
        scores = []
        for path_gen in gen_list:
            #print(path_gen)
            gen.load_state_dict(torch.load(path_gen,map_location='cuda:'+str(parser.device)))
            gen.eval()
            #print('construct dataset...')
            if 'cifar' in dataset:
                X, Y = return_x_y_filtered(gen,classifier,1000,1000*100,z_dim,parser.device)
            else:
                X, Y = return_x_y(gen,classifier,1000,100,z_dim,parser.device)
            X_test, Y_test = return_x_y_filtered(gen,classifier,1000,1000*10,z_dim,parser.device)
            #print('training...')
            clf = LogisticRegression(penalty='none',solver='lbfgs',verbose=0).fit(X, Y)
            #print('testing...')
            sc = clf.score(X_test,Y_test)
            scores.append(sc*100)
            #print(sc)
        print_format_metric(scores)
    
    print('Distribution fitting')
    batch_size = 250
    n_iter = 200
    if dataset == 'mnist':
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
    elif dataset == 'cifar10':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    elif dataset == 'synthetic_cifar10':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        train_data = datasets.ImageFolder(root='data/synthetic_cifar10/', transform=transform)
        test_data = datasets.ImageFolder(root='data/synthetic_cifar10/', transform=transform)
    elif dataset == 'cifar100':
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        train_data = datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=batch_size, 
                                               shuffle=True, 
                                              num_workers=1) 
    test_loader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              num_workers=1)
    dataiter = iter(train_loader)

    prec, rec, dens, cov, fids = [],[],[],[],[]
    for path_gen in gen_list:
        #print(path_gen)
        gen.load_state_dict(torch.load(path_gen,map_location='cuda:'+str(parser.device)))
        gen.eval()
        X_gen = return_gen_features(gen,classifier,batch_size,n_iter,z_dim,parser.device).cpu().numpy()
        X_real = return_real_features(classifier,n_iter,dataiter,train_loader,parser.device).cpu().numpy()
        mu1, sigma1 = get_mu_sigma(X_real)
        mu2, sigma2 = get_mu_sigma(X_gen)
        fid = calculate_frechet_distance(mu1,sigma1,mu2,sigma2)
        dict_prdc = compute_prdc(X_gen[:10000],X_real[:10000],nearest_k=5)
        prec.append(dict_prdc['precision']*100)
        rec.append(dict_prdc['recall']*100)
        dens.append(dict_prdc['density']*100)
        cov.append(dict_prdc['coverage']*100)
        fids.append(fid)
        #print(dict_prdc,fid)

    print_format_metric(fids)
    print_format_metric(prec)
    print_format_metric(rec)
    print_format_metric(dens)
    print_format_metric(cov)
        
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()