#!/bin/bash
#SBATCH --job-name=baseline_extract
#SBATCH --output=logs/baseline_extract_%j.out
#SBATCH --error=logs/baseline_extract_%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=64

# Set up environment
cd $SCRATCH/autotslabel
source .venv/bin/activate

# Run training
srun python -m autotslabel.autosegment.multichannel.step_2b_filter_spectrogram