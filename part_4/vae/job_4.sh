#!/bin/bash
#SBATCH --job-name=part4_vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=part4_vae_%j.out
#SBATCH --error=part4_vae_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python part4.py
