# Echocardiogram video data repository

## Introduction

First, the repository requires:

1. `~/.venv` in root directory with dependencies installed.
2. Unzipped `/EchoNet-Dynamic/` in `~/datasets`.
    - Available @ [EchoNet-Dynamic](https://www.kaggle.com/datasets/mahnurrahman/echonet-dynamic)

## Contents

1. `~/simclr` contains the remade contents of the original SimCLR repository.
2. `echonet_eda.ipynb` contains the minor exploratory data analysis contents & prototyping.

## Running code

Example command with required arguments are:

```python
python run.py -a vit_b_16 -data ../datasets/EchoNet-Dynamic/
```

which just specifies the model architecture and the dataset root path.

There are other flags specified in `run.py`.
