import argparse
import datetime
import logging
import math
from pathlib import Path

from accelerate import Accelerator, InitProcessGroupKwargs
from timm.optim import Lars
import torch

from models.model import get_model
from optimizer import get_optimizer
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
    choices=["vit", "vivit", "resnet50"],
    dest="arch",
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
parser.add_argument(
    '-o', '--optimizer',
    default='lars',
    choices=['lars', 'adam'],
    help='optimizer to use (rec. to use lars for >> batch sizes)'
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
    help='log to tensorboard every n steps'
)
parser.add_argument(
    '--save-every-n',
    default=1,
    type=int,
    help='saves the model every n epochs'
)
parser.add_argument(
    '--save-last-n',
    default=10,
    type=int,
    help='saves the model for the last n epochs'
)

def main():
    process_group_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=1))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

    args = parser.parse_args()
    args.date = datetime.datetime.now()

    if accelerator.is_main_process:
        setup_logging(Path("./logs/pretraining"), model_arch=args.arch, date=args.date)

    with accelerator.main_process_first():
        dataset = ContrastiveLearningDataset(
            root_folder=args.data,
            model_type=args.arch,
            dataset_name=args.dataset_name,
            addl_args=args
        )

    train_data = dataset.get_dataset_split(dataset_name=args.dataset_name, split="train")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    model = get_model(
        model_name=args.arch,
        dataset_name=args.dataset_name,
        clip_length=args.clip_length,
        finetune_mode=False
    )

    optimizer, scheduler = get_optimizer(model, len(train_loader), args)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    if accelerator.is_main_process:
        logging.info(f"Starting training with {args.arch} model.")
    
    simclr_model = SimCLR(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        accelerator=accelerator, 
        args=args
    )
    simclr_model.train(train_loader)

if __name__ == "__main__":
    main()