# Echocardiogram video data repository

## Introduction

Note that this is done on Python 3.9.6. Upgraded versions should work too.

First, the repository requires:

1. `~/.venv` in root directory with dependencies installed.
2. Unzipped `/EchoNet-Dynamic/` in `~/datasets`.
    - Available @ [EchoNet-Dynamic](https://www.kaggle.com/datasets/mahnurrahman/echonet-dynamic)

## Contents

1. `~/simclr` contains the remade contents of the original SimCLR repository.
2. `echonet_eda.ipynb` contains the minor exploratory data analysis contents & prototyping.

## Running code

Example command with required arguments:

```bash
python run.py -a vit_b_16 -data ../datasets/EchoNet-Dynamic/
```

This specifies the model architecture and the dataset root path. Below is a full list of available command-line flags and their descriptions.

### Command-line arguments

- `-a`, `--arch` (`ARCH`)

    Model architecture to use. Must be one of the keys defined in `model_dict`.

    **Example:** `vit_b_16`

- `-data` (`DIR`)

    Path to the dataset root directory.

    **Example:** `../datasets/EchoNet-Dynamic/`

- `-dataset-name`

    Name of the dataset. Used to select dataset-specific loading logic.

    **Choices:** `echonet-dynamic`, `cifar10`
    **Default:** `echonet-dynamic`

- `--lr`, `--learning-rate` (`LR`)

    Initial learning rate.

    **Default:** `0.0003`

- `--wd`, `--weight-decay` (`W`)

    Weight decay (L2 regularization strength).

    **Default:** `1e-4`

- `-b`, `--batch-size` (`N`)

    Mini-batch size. This is the **total batch size across all GPUs** when using Data Parallel or Distributed Data Parallel.

    **Default:** `256`

- `--epochs` (`N`)

    Total number of training epochs.

    **Default:** `200`

- `-j`, `--workers` (`N`)

    Number of data loading worker processes.

    **Default:** `12`

- `--temperature`

    Softmax temperature used in the contrastive loss.

    **Default:** `0.07`

- `--log-every-n-steps`

    Log training metrics every _n_ steps.

    **Default:** `100`

- `--gpu-index`

    GPU index to use when CUDA is enabled.

    **Default:** `0`

- `--disable-cuda`

    Disable CUDA and force CPU-only training.

- `--fp16-precision`

    Enable mixed-precision (16-bit / FP16) training on GPU.

- `--clip-length`

    Lets users decide clip length when training with ViViT.

### Example with additional flags

```bash
python run.py \
  -a vit_b_16 \
  -data ../datasets/EchoNet-Dynamic/ \
  --epochs 100 \
  -b 128 \
  --lr 3e-4 \
  --temperature 0.1 \
  --fp16-precision
```

files that need to be modified to add a new model

format of the dataset (dict, train, test, val) in order to use other ones

command to evaluate model
