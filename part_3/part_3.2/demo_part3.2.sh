#!/bin/bash
#SBATCH --job-name=demo_3.2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=demo_3.2_%j.out
#SBATCH --error=demo_3.2_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python demo_part3.2.py
