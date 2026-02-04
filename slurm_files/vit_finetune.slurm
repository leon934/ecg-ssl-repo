#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

data_path=./datasets/EchoNet-Dynamic/
model_path=./models/pretraining/02-03-2026_22:45:52-vit-checkpoints/checkpoint_0799.pth.tar
batch_size=512
num_epochs=100
num_workers=8

python echo_simclr/run_finetune.py \
	--model-path ${model_path} \
	--data ${data_path} \
	--batch-size ${batch_size} \
	--epochs ${num_epochs} \
	--workers ${num_workers} \
	--fp16-precision \
    --save-every-n 5 \
    --save-last-n 10