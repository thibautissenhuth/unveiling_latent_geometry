from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import cv2
import torchvision

#from utils.fid_score import calculate_fid_given_paths
# from utils.torch_fid_score import get_fid
# from utils.inception_score import get_inception_score
from functions import validate

logger = logging.getLogger(__name__)

import cfg
import models_search
from functions import validate
from utils.utils import set_log_dir, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from utils.inception_score import get_inception_score

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
#     assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir('logs_eval', args.exp_name)
    logger = create_logger(args.path_helper['log_path'], phase='test')

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models_search.'+args.gen_model+'.Generator')(args=args).cuda()
    gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=[0])

    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (4, args.latent_dim)))

    # set writer
    logger.info(f'=> resuming from {args.load_path}')
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    print(gen_net)

    if 'avg_gen_state_dict' in checkpoint:
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        epoch = checkpoint['epoch']
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f'=> loaded checkpoint {checkpoint_file}')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': 0,
    }

        # Generate a batch of images
    with torch.no_grad():
        os.makedirs('samples/'+args.exp_name,exist_ok=True)
        if args.simplicial_sampling:
            gen_imgs_acc =  gen_net(z, epoch,rejected=False)
            gen_imgs_rej =  gen_net(z, epoch,rejected=True)
            gen_imgs = torch.cat((gen_imgs_acc[:50],gen_imgs_rej[:50]),dim=0)
            torchvision.utils.save_image(gen_imgs, 'samples/'+args.exp_name+'/generated_set_acc_rej.png',normalize=True,nrow=10)
            torchvision.utils.save_image(gen_imgs_acc, 'samples/'+args.exp_name+'/generated_set_acc.png',normalize=True,nrow=10)
            torchvision.utils.save_image(gen_imgs_rej, 'samples/'+args.exp_name+'/generated_set_rej.png',normalize=True,nrow=10)
        else:
            gen_imgs = gen_net(z, epoch)
        torchvision.utils.save_image(gen_imgs[:100], 'samples/'+args.exp_name+'/generated_set.png',normalize=True)
    print("image saved")

    inception_score, fid_score = validate(args, fixed_z, None, epoch, gen_net, writer_dict, clean_dir=False)
    logger.info(f'Inception score: {inception_score}, FID score: {fid_score}.')


if __name__ == '__main__':
    main()
