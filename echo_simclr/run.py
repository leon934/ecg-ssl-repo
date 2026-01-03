import argparse
import logging
from pathlib import Path
import os

import torch

from model import ViTModel, model_dict
from data_aug.cl_dataset import ContrastiveLearningDataset
from simclr import SimCLR

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('-a', '--arch', metavar='ARCH',
                    choices=model_dict,
                    help='model architecture: ' +
                         ' | '.join(model_dict))
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='echonet-dynamic',
                    help='dataset name', choices=['echonet-dynamic'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--gpu-index', default=0, type=int,
                    help='GPU index to use (default: 0)')

def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "training.log",
        level=logging.DEBUG,
        filemode="w"
    )

def main():
    args = parser.parse_args()

    args.device = (
        f"cuda:{args.gpu_index}"
        if torch.cuda.is_available() and args.gpu_index is not None and args.gpu_index >= 0
        else "cpu"
    )
    args.workers = max(1, os.cpu_count() // 2)
    args.model_dict = model_dict

    data = ContrastiveLearningDataset(args.data)
    dataset = data.get_dataset(args.dataset_name, args.arch, args.model_dict)

    train_data = dataset["TRAIN"]
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    model = ViTModel(base_model=args.arch)
    model = model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, 
                                                           last_epoch=-1)
    
    simclr_model = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr_model.train(train_loader)

if __name__ == "__main__":
    setup_logging(Path("./logs"))
    main()