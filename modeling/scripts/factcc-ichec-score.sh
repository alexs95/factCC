#!/bin/sh
#SBATCH -p GpuQ
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH -A project
#SBATCH --mail-user=email@domain.com
#SBATCH --mail-type=BEGIN,END

module load cuda/11.2
module load conda/2

conda init
source activate summarization3.7

cd $SLURM_SUBMIT_DIR
time python modeling/score.py --mode evaluate