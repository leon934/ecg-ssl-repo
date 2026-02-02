#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

data_path=./datasets/EchoNet-Dynamic/
batch_size=16
num_epochs=200
model=vivit
clip_length=16

python echo_simclr/run_pretrain.py \
	--model ${model} \
	--data ${data_path} \
	--batch-size ${batch_size} \
	--epochs ${num_epochs} \
	--workers 6 \
	--fp16-precision \
	--clip-length ${clip_length} \
    --save-every-n 5 \
    --save-last-n 10