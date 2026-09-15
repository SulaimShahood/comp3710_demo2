#!/bin/bash
#SBATCH --job-name=part3.2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=part3.2_%j.out
#SBATCH --error=part3.2_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python part3.2.py
