#!/usr/bin/env bash
#SBATCH --job-name=Sim2Real-MotionX
#SBATCH --output=/mimer/NOBACKUP/groups/focs/slurm/logs/%x/%j/out.log
#SBATCH --error=/mimer/NOBACKUP/groups/focs/slurm/logs/%x/%j/err.log
#SBATCH -A NAISS2025-22-332 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:4
#SBATCH -t 5-00:00:00



module load Python/3.11.5-GCCcore-13.2.0
source /mimer/NOBACKUP/groups/focs/virtualenv/real2sim/bin/activate


torchrun --standalone --nproc_per_node=4 /cephyr/users/fracal/Alvis/Real2Sim/Pretraining/multigpu.py
wait
