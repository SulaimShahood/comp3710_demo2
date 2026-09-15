#!/bin/bash
#SBATCH --job-name= JOB NAME
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=JOB NAME_%j.out
#SBATCH --error=JOB NAME_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

python <name_of_python_file_to_run>
