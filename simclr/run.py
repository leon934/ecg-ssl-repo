import torch
from tqdm import tqdm

import argparse

from model import ViTModel, vit_dict
from data_aug.cl_dataset import ContrastiveLearningDataset

parser = argparse.ArgumentParser(description='PyTorch SimCLR')

parser.add_argument('-a', '--arch', metavar='ARCH',
                    choices=vit_dict,
                    help='model architecture: ' +
                         ' | '.join(vit_dict))
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
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

def main():
    tqdm.write("Parsing arguments...")
    args = parser.parse_args()

    device = (
        f"cuda:{args.gpu_index}"
        if torch.cuda.is_available() and args.gpu_index is not None and args.gpu_index >= 0
        else "cpu"
    )

    tqdm.write("Loading dataset...")
    data = ContrastiveLearningDataset(args.data)
    dataset = data.get_dataset(args.dataset_name)

    tqdm.write("Placing dataset into data loader...")
    train_data = dataset["TRAIN"]
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    tqdm.write("Creating model...")
    model = ViTModel(base_model=args.arch)
    model = model.to(device)
    
    tqdm.write("Creating optimizer and scheduler...")
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, 
                                                           last_epoch=-1)
    

if __name__ == "__main__":
    main()