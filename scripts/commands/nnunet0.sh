#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --output=logs/nnunet_%j.out
#SBATCH --error=logs/nnunet_%j.err
#SBATCH --time=24:00:00
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

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 2 2d 0 -p nnUNetResEncUNetLPlans --npz --c