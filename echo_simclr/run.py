import argparse
import logging
from pathlib import Path
import datetime

import torch

from models.model import get_model
from models.vit.model import model_dict
from cl_dataset import ContrastiveLearningDataset
from simclr import SimCLR

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
parser.add_argument(
    '-a', '--arch',
    metavar='ARCH',
    choices=model_dict.keys(),
    default=None,
    help='model architecture: ' + ' | '.join(model_dict)
)

# -------------------------------------------------
# Dataset
# -------------------------------------------------
parser.add_argument(
    '-data',
    metavar='DIR',
    help='path to dataset',
    required=True
)
parser.add_argument(
    '-dataset-name',
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
    '--gpu-index',
    default=0,
    type=int,
    help='GPU index to use (default: 0)'
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


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "training-{date:%m-%d-%Y_%H:%M:%S}.log".format(date=datetime.datetime.now()),
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
    args.model_dict = model_dict

    data = ContrastiveLearningDataset(args.data)
    dataset = data.get_dataset(args.dataset_name, args)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    model = get_model(model_name=args.model, dataset_name=args.dataset_name, clip_length=args.clip_length, arch_type=args.arch)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs, eta_min=0, 
                                                           last_epoch=-1)
    
    logging.info(f"Starting training with {args.model} model.")
    
    simclr_model = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr_model.train(train_loader)

if __name__ == "__main__":
    setup_logging(Path("./logs"))
    main()