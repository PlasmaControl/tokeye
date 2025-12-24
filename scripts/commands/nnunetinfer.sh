#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --output=logs/nnunet_%j.out
#SBATCH --error=logs/nnunet_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3

# Set up environment
cd $SCRATCH/autotslabel
source .venv/bin/activate

export nnUNet_raw="/scratch/gpfs/nc1514/autotslabel/data/cache/nnunet/raw"
export nnUNet_preprocessed="/scratch/gpfs/nc1514/autotslabel/data/cache/nnunet/preprocessed"
export nnUNet_results="/scratch/gpfs/nc1514/autotslabel/data/cache/nnunet/results"

nnUNetv2_predict -d Dataset002_ECEBinary -i /scratch/gpfs/nc1514/autotslabel/notebooks/180634_spec_input/window_0000 -o /scratch/gpfs/nc1514/autotslabel/notebooks/180634_spec_output/window_0000 -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetResEncUNetLPlans