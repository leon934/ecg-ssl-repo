#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:t4:4
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

data_path=./datasets/EchoNet-Dynamic/
batch_size=64
num_epochs=200
model=vivit
clip_length=16

accelerate launch \
    --num_processes=4 \
    echo_simclr/run.py \
	-m ${model} \
	-data ${data_path} \
	-b ${batch_size} \
	--epochs ${num_epochs} \
	--gpu-index 0 \
	-j 6 \
	--fp16-precision \
	--clip-length ${clip_length}
