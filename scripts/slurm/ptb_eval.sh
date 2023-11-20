#!/usr/bin/env bash
#SBATCH --account=jonmay_231
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-task=v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=outputs/ptb_eval_output.txt
#SBATCH --error=outputs/ptb_eval_error.txt
#SBATCH --mail-type=BEGIN,FAIL,END        # send email when job begins
#SBATCH --mail-user=zhejianz@usc.edu

source scripts/slurm/module.sh
source scripts/slurm/cpu.sh
source scripts/slurm/conda.sh
bash scripts/ptb_eval.sh
