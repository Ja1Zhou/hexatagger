#!/usr/bin/env bash
#SBATCH --account=jonmay_231
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=outputs/infer_output.txt
#SBATCH --error=outputs/infer_error.txt

source scripts/slurm/module.sh
source scripts/slurm/cpu.sh
source scripts/slurm/conda.sh
bash scripts/ptb_infer.sh
