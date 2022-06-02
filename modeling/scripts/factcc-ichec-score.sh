#!/bin/sh
#SBATCH -p GpuQ
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH -A ngcom023c
#SBATCH --mail-user=a.shapovalov1@nuigalway.ie
#SBATCH --mail-type=ALL

module load cuda/11.2
module load conda/2

conda init
source activate summarization3.6

cd $SLURM_SUBMIT_DIR
for d in $PWD/evaluation/*/ ; do
    echo "$d"
    if [[ $d == *"paragraph"* ]]; then
	python3 modeling/score.py --mode evaluate --evaluation $d --gpu --paragraph
    else
	python3 modeling/score.py --mode evaluate --evaluation $d --gpu
    fi
    echo ""
done

