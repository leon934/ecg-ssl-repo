#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

data_path=./datasets/EchoNet-Dynamic/
model=vit
batch_size=64
num_epochs=200
num_workers=4

accelerate launch \
    --num_processes=2 \
    --mixed_precision=fp16 \
    --num_machines=1 \
    --dynamo_backend=no \
    echo_simclr/run_pretrain.py \
	--model ${model} \
	--data ${data_path} \
	--batch-size ${batch_size} \
	--epochs ${num_epochs} \
	--workers ${num_workers} \
	--fp16-precision