#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu-a40

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

data_path=./datasets/EchoNet-Dynamic/
model=vit
batch_size=256
num_epochs=300
num_workers=2

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
    --save-every-n 5 \
    --save-last-n 10