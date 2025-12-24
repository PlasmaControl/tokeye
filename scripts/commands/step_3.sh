#!/bin/bash
#SBATCH --job-name=correlation_analysis
#SBATCH --output=logs/step_3_correlation_analysis_%j.out
#SBATCH --error=logs/step_3_correlation_analysis_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Set up environment
cd $SCRATCH/autotslabel
source .venv/bin/activate

# Run training
srun python -m autotslabel.autosegment.multichannel.step_3a_correlation_analysis
srun python -m autotslabel.autosegment.multichannel.step_3b_extract_correlation