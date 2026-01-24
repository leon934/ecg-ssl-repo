# Echocardiogram video data repository

## Introduction

Note that this is done on Python 3.9.6. Upgraded versions should work too.

First, the repository requires:

1. `~/.venv` in root directory with dependencies installed.
2. Unzipped `/EchoNet-Dynamic/` in `~/datasets`.
    - Available @ [EchoNet-Dynamic](https://www.kaggle.com/datasets/mahnurrahman/echonet-dynamic)

After unzipping the file and adding the contents to `~/datasets/`, you should have `~/datasets/EchoNet-Dynamic/` which you will use to point the process to the correct dataset.

## Contents

1. `~/simclr` contains the remade contents of the original SimCLR repository.
2. `echonet_eda.ipynb` contains the minor exploratory data analysis contents & prototyping.

## Running code for pretraining the model

The `*_train.slurm` files can be viewed to get a good idea on how to train each of the models.

Below is a full list of available command-line flags and their descriptions.

### Command-line arguments

---

### Model / Architecture

| Argument        | Type   | Choices / Default          | Description               | Required |
| --------------- | ------ | -------------------------- | ------------------------- | -------- |
| `-m`, `--model` | string | `vit`, `vivit`, `resnet50` | Model architecture to use | Yes      |

---

### Dataset

| Argument         | Type   | Choices / Default | Description               | Required |
| ---------------- | ------ | ----------------- | ------------------------- | -------- |
| `--data`         | string | —                 | Path to dataset directory | Yes      |
| `--dataset-name` | string | `echonet-dynamic` | Name of the dataset       | No       |

---

### Training

| Argument             | Type  | Default | Description                            | Required |
| -------------------- | ----- | ------- | -------------------------------------- | -------- |
| `-b`, `--batch-size` | int   | 256     | Mini-batch size; total across all GPUs | No       |
| `--epochs`           | int   | 200     | Number of total epochs to run          | No       |
| `--temperature`      | float | 0.07    | Softmax temperature                    | No       |
| `--clip-length`      | int   | 32      | Clip length for ViViT                  | No       |

---

### Optimization

| Argument                  | Type  | Default | Description           | Required |
| ------------------------- | ----- | ------- | --------------------- | -------- |
| `--lr`, `--learning-rate` | float | 3e-4    | Initial learning rate | No       |
| `--wd`, `--weight-decay`  | float | 1e-4    | Weight decay          | No       |

---

### System / Hardware

| Argument          | Type | Default | Description                    | Required |
| ----------------- | ---- | ------- | ------------------------------ | -------- |
| `--disable-cuda`  | flag | False   | Disable CUDA                   | No       |
| `-j`, `--workers` | int  | 12      | Number of data loading workers | No       |

---

### Logging

| Argument              | Type | Default | Description                          | Required |
| --------------------- | ---- | ------- | ------------------------------------ | -------- |
| `--log-every-n-steps` | int  | 100     | Log to TensorBoard every n steps     | No       |
| `--save-every-n`      | int  | 1       | Save model every n epochs            | No       |
| `--save-last-n`       | int  | 10      | Save the model for the last n epochs | No       |

---

### Precision

| Argument           | Type | Default | Description                           | Required |
| ------------------ | ---- | ------- | ------------------------------------- | -------- |
| `--fp16-precision` | flag | False   | Use 16-bit (mixed) precision training | No       |
---

## Running code for fine-tuning the model

The `*_fientune.slurm` files can be viewed to get a good idea on how to train each of the models.

Below is a full list of available command-line flags and their descriptions.

---

### Finetuning

| Argument              | Type   | Choices / Default  | Description                            | Required |
| --------------------- | ------ | ------------------ | -------------------------------------- | -------- |
| `-mp`, `--model-path` | string | —                  | Path to `.pth` model file              | Yes      |
| `-b`, `--batch-size`  | int    | 256                | Mini-batch size; total across all GPUs | No       |
| `--epochs`            | int    | 200                | Number of total epochs to run          | No       |
| `-y`, `--y-name`      | string | `EF`, `EDV`, `ESV` | Target variable                        | Yes      |

---

### Dataset

| Argument         | Type   | Choices / Default | Description                 | Required |
| ---------------- | ------ | ----------------- | --------------------------- | -------- |
| `--data`         | string | —                 | Path to dataset root folder | Yes      |
| `--dataset-name` | string | `echonet-dynamic` | Dataset name                | No       |

---

### Optimization

| Argument                  | Type  | Choices / Default | Description           | Required |
| ------------------------- | ----- | ----------------- | --------------------- | -------- |
| `--lr`, `--learning-rate` | float | 3e-4              | Initial learning rate | No       |
| `--wd`, `--weight-decay`  | float | 1e-4              | Weight decay          | No       |

---

### System / Hardware

| Argument          | Type | Choices / Default | Description                    | Required |
| ----------------- | ---- | ----------------- | ------------------------------ | -------- |
| `-j`, `--workers` | int  | 12                | Number of data loading workers | No       |

---

### Logging

| Argument              | Type | Choices / Default | Description                      | Required |
| --------------------- | ---- | ----------------- | -------------------------------- | -------- |
| `--log-every-n-steps` | int  | 100               | Log every n steps                | No       |
| `--save-every-n`      | int  | 1                 | Save model every n epochs        | No       |
| `--save-last-n`       | int  | 10                | Save model for the last n epochs | No       |

---

### Precision

| Argument           | Type | Choices / Default | Description                           | Required |
| ------------------ | ---- | ----------------- | ------------------------------------- | -------- |
| `--fp16-precision` | flag | False             | Use 16-bit (mixed) precision training | No       |

---