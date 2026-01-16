#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=15:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
source .venv/bin/activate

data_path=./datasets/EchoNet-Dynamic/
model=vit
model_arch=vit_b_16
batch_size=256
num_epochs=200

python echo_simclr/run.py \
	--model ${model} \
	--arch ${model_arch} \
	-data ${data_path} \
	-b ${batch_size} \
	--epochs ${num_epochs} \
	--gpu-index 0 \
	-j 12 \
	--fp16-precision
