import argparse
import datetime
from pathlib import Path

from accelerate import Accelerator
import torch

from cl_dataset import ContrastiveLearningDataset
from models.vit.model import ViTModel
from models.vivit.model import ViViTModel
from models.resnet50.model import ResNetModel
from simclr import SimCLR
from utils import setup_logging

parser = argparse.ArgumentParser(description='PyTorch SimCLR evaluation pipeline')

# -------------------------------------------------
# Finetuning
# -------------------------------------------------
parser.add_argument(
    '-mp', '--model-path',
    metavar="MODEL_PATH",
    required=True,
    help='path to .pth model file'
)
parser.add_argument(
    '--epochs',
    default=200,
    type=int,
    metavar='N',
    help='number of total epochs to run'
)
parser.add_argument(
    '-b', '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256); total batch size across all GPUs'
)
# parser.add_argument(
#     '-y', '--y-name',
#     metavar='y_name',
#     required=True,
#     choices=["EF", "EDV", "ESV"],
#     help='target variable, choices are EF, EDV, and ESV'
# )
# -------------------------------------------------
# Dataset
# -------------------------------------------------
parser.add_argument(
    '--data',
    metavar='DIR',
    help='path to dataset root folder',
    required=True
)
parser.add_argument(
    '--dataset-name',
    default='echonet-dynamic',
    choices=['echonet-dynamic'],
    help='dataset name'
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
    default='nestorov',
    choices=['lars', 'adam', 'nestorov'],
    help='optimizer to use (rec. to use lars for >> batch sizes)'
)
# -------------------------------------------------
# System / Hardware
# -------------------------------------------------
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
parser.add_argument(
    '--fp16-precision',
    action='store_true',
    help='enables fp16 precision to reduce memory load'
)

def main():
    args = parser.parse_args()
    args.date = datetime.datetime.now()

    model_data = torch.load(args.model_path, map_location="cpu")
    args.arch, model_state_dict = model_data["model"], model_data["state_dict"]

    setup_logging(Path("./logs/finetuning"), model_arch=args.arch, date=args.date)

    dataset = ContrastiveLearningDataset(
        root_folder=args.data,
        model_type=args.arch,
        dataset_name=args.dataset_name,
        addl_args=args
    )
    dataloader_dict = {f"{split}_loader": torch.utils.data.DataLoader(
        dataset.get_dataset_split(args.dataset_name, split, use_Y=True),
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=(split == "train")
    ) for split in ("train", "test", "val")}

    # gets backbone w/o projection head
    model_state_dict = {layer : weights for layer, weights in model_state_dict.items() if not layer.startswith("head")}
    
    # lambdas are necessary so unseen configs aren't ran until model key is called
    curr_model_params = {
        "vivit": (
            lambda **kwargs: ViViTModel(**kwargs), 
            {
                "clip_length": lambda: model_data["config"]["num_frames"],
            }
        ),
        "vit": (
            lambda **kwargs: ViTModel(**kwargs),
            {}
        ),
        "resnet50": (
            lambda **kwargs: ResNetModel(**kwargs),
            {}
        )
    }

    ModelClass, addl_model_args = curr_model_params[args.arch]
    # strict=False bcos we now have new randomly initialized head attached to model
    model = ModelClass(
        image_size=112,
        channels=1,
        finetune_mode=True,
        **{param: arg() for param, arg in addl_model_args.items()}
    )
    model.load_state_dict(model_state_dict, strict=False)
    model = model.float()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # no scheduler/weight decay mentioned in simclr paper for ft
    simclr_model = SimCLR(
        model=model, 
        optimizer=optimizer,
        scheduler=None,
        accelerator=None,
        args=args
    )
    simclr_model.finetune(**dataloader_dict)

if __name__ == "__main__":
    main()