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

# control cpu threads
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16

bash scripts/ptb.sh