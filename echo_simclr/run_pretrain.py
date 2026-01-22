import argparse
import logging
from pathlib import Path

from accelerate import Accelerator
import torch

from models.model import get_model
from cl_dataset import ContrastiveLearningDataset
from simclr import SimCLR
from utils import setup_logging

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

# -------------------------------------------------
# Model / Architecture
# -------------------------------------------------
parser.add_argument(
    '-m', '--model',
    metavar="MODEL",
    choices=["vit", "vivit"],
    help='available models: vivit and vit',
    required=True
)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
parser.add_argument(
    '--data',
    metavar='DIR',
    help='path to dataset directory',
    required=True
)
parser.add_argument(
    '--dataset-name',
    default='echonet-dynamic',
    choices=['echonet-dynamic'],
    help='dataset name'
)

# -------------------------------------------------
# Training
# -------------------------------------------------
parser.add_argument(
    '-b', '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256); total batch size across all GPUs'
)
parser.add_argument(
    '--epochs',
    default=200,
    type=int,
    metavar='N',
    help='number of total epochs to run'
)
parser.add_argument(
    '--temperature',
    default=0.07,
    type=float,
    help='softmax temperature (default: 0.07)'
)
parser.add_argument(
    '--clip-length',
    default=32,
    type=int,
    help='clip length for vivit (default: 32)'
)

# -------------------------------------------------
# Optimization
# -------------------------------------------------
parser.add_argument(
    '--lr', '--learning-rate',
    default=3e-4,
    type=float,
    metavar='LR',
    dest='lr',
    help='initial learning rate'
)
parser.add_argument(
    '--wd', '--weight-decay',
    default=1e-4,
    type=float,
    metavar='W',
    dest='weight_decay',
    help='weight decay (default: 1e-4)'
)

# -------------------------------------------------
# System / Hardware
# -------------------------------------------------
parser.add_argument(
    '--disable-cuda',
    action='store_true',
    help='disable CUDA'
)
parser.add_argument(
    '-j', '--workers',
    default=12,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 12)'
)
# -------------------------------------------------
# Logging
# -------------------------------------------------
parser.add_argument(
    '--log-every-n-steps',
    default=100,
    type=int,
    help='log every n steps'
)
# -------------------------------------------------
# Precision
# -------------------------------------------------
parser.add_argument(
    '--fp16-precision',
    action='store_true',
    help='use 16-bit (mixed) precision training'
)

def main():
    args = parser.parse_args()

    # moves to gpu if possible when fp16 flag enabled
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16_precision else "no"
    )

    dataset = ContrastiveLearningDataset(
        root_folder=args.data,
        model_type=args.model,
        dataset_name=args.dataset_name,
        addl_args=args
    )

    train_data = dataset.get_dataset_split(dataset_name=args.dataset_name, split="TRAIN")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    model = get_model(model_name=args.model, dataset_name=args.dataset_name, clip_length=args.clip_length)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=0, 
                                                           last_epoch=-1)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    logging.info(f"Starting training with {args.model} model.")
    
    simclr_model = SimCLR(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        accelerator=accelerator, 
        args=args
    )
    simclr_model.train(train_loader)

if __name__ == "__main__":
    setup_logging(Path("./logs/pretraining"),)
    main()