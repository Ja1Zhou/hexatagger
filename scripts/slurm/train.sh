#!/bin/bash
#SBATCH --account=<jonmay_231>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
module purge
module load gcc
module load conda
module load cuda/11.8.0

source scripts/slurm/cpu.sh
bash scripts/ptb.sh