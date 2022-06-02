#!/bin/sh
#SBATCH -p DevQ
#SBATCH --nodes 1
#SBATCH --time 01:00:00
#SBATCH -A ngcom023c
#SBATCH --mail-user=a.shapovalov1@nuigalway.ie
#SBATCH --mail-type=ALL

module load taskfarm
module load conda/2

conda init
source activate summarization3.6

cd $SLURM_SUBMIT_DIR
taskfarm modeling/scripts/factcc-ichec-score-taskfarm.sh
