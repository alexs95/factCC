#!/bin/sh
#SBATCH -p GpuQ
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -A account

cd $SLURM_SUBMIT_DIR

module load cuda/11.2
module load conda/2

source activate summarization3.7

echo "This is the GpuQ run."
time python modeling/score.py --mode evaluate