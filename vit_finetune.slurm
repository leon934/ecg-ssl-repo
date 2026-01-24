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
model_path=./models/01-23-2026_03:51:31_checkpoints/checkpoint_0499.pth.tar
batch_size=256
num_epochs=500
num_workers=8

python echo_simclr/run_finetune.py \
	--model-path ${model_path} \
	--data ${data_path} \
	--batch-size ${batch_size} \
    --y-name EF \
	--epochs ${num_epochs} \
	--workers ${num_workers} \
	--fp16-precision \
    --save-every-n 5 \
    --save-last-n 10