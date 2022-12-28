import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from networks.vision_transformer import SwinUnet as ViT_seg
from networks.swin_v2 import unet_swin
from config import get_config
from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader
from PIL import Image

pred_path = '/kaggle/working/output/pred'
os.makedirs(pred_path, exist_ok=True)
gt_path = '/kaggle/working/output/gt'
os.makedirs(gt_path, exist_ok=True)
# edg_path = '/kaggle/working/output/edg'
# os.makedirs(edg_path, exist_ok=True)
# fus_path = '/kaggle/working/output/fus'
# os.makedirs(fus_path, exist_ok=True)
def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    n_val = len(loader)
    pred_idx=0
#     edg_idx=0
#     fus_idx=0
    gt_idx=0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            
            trainsize = 512 #rate        
            imgs = F.interpolate(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            true_masks = F.interpolate(true_masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.2).float()
            
#             edg = torch.sigmoid(edge)
#             edg = (edg > 0.2).float()
#             fus = torch.sigmoid(fused)
#             fus = (fus > 0.2).float()
            
            for img in pred:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(pred_path+'/'+str(pred_idx)+'.png')
                pred_idx += 1
                
#             for img in edg:
#                 img = img.squeeze(0).cpu().numpy()
#                 img = Image.fromarray((img * 255).astype(np.uint8))
#                 img.save(edg_path+'/'+str(edg_idx)+'.png')
#                 edg_idx += 1
#             for img in pred:
#                 img = img.squeeze(0).cpu().numpy()
#                 img = Image.fromarray((img * 255).astype(np.uint8))
#                 img.save(fus_path+'/'+str(fus_idx)+'.png')
#                 fus_idx += 1
                
            for img in true_masks:
                img = img.squeeze(0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(gt_path+'/'+str(gt_idx)+'.png')
                gt_idx += 1

            pbar.update()


def test_net(net,
              device,
              batch_size=1,
              n_class=1,
              img_size=448):


    val_img_dir = '/kaggle/working/datasets/test/images/'
    val_mask_dir = '/kaggle/working/datasets/test/masks/'

    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    net.eval()

    eval_net(net, val_loader, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=448,
                        help='The size of the images')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224_lite.yaml', 
                        required=False, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    #config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # net = UNet(128, 1)
    # net = nn.DataParallel(net, device_ids=[0])
    # net.to(device=device)
    
    #net = ViT_seg(config, img_size=args.size, num_classes=1).cuda()
    net = unet_swin(img_size=512,size="swinv2_tiny_window8_256").cuda()
    
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device), False
        )
        logging.info(f'Model loaded from {args.load}')

    try:
        test_net(net=net,
                  batch_size=args.batchsize,
                  device=device,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
