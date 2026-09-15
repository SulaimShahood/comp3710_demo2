#!/bin/bash
#SBATCH --job-name=part3.1
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=part3.1_%j.out
#SBATCH --error=part3.1_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python part3.1.py
