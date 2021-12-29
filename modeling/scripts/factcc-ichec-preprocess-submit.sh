#!/bin/sh
#SBATCH -p GpuQ
#SBATCH --nodes 1
#SBATCH --time 00:10:00
#SBATCH -A project
#SBATCH --mail-user=email@domain.com
#SBATCH --mail-type=BEGIN,END

module load conda/2

conda init
source activate summarization3.6

cd $SLURM_SUBMIT_DIR
taskfarm factcc-ichec-preprocess-taskfarm.sh
