#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

model=vit
data_path=./datasets/EchoNet-Dynamic/
batch_size=512
num_epochs=800
num_workers=2
optimizer=lars

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
    --optimizer ${optimizer} \
    --save-every-n 40 \
    --save-last-n 25