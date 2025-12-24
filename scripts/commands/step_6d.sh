#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH --cpus-per-task=6

# Set up environment
cd $SCRATCH/autotslabel
source .venv/bin/activate

# Run training
srun python -m autotslabel.autosegment.multichannel.step_6d_final