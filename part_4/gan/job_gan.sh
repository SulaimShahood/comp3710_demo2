#!/bin/bash
#SBATCH --job-name=part4_gan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=gan_%j.out
#SBATCH --error=gan_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python part4_gan.py
