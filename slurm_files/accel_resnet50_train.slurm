#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

model=resnet50
data_path=./datasets/EchoNet-Dynamic/
batch_size=512
num_epochs=200

accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    --num_machines=1 \
    --dynamo_backend=no \
    echo_simclr/run_pretrain.py \
	-m ${model} \
	--data ${data_path} \
	-b ${batch_size} \
	--epochs ${num_epochs} \
	-j 4 \
    --save-every-n 5 \
    --save-last-n 10