#!/bin/sh
#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=c12m85-a100-2
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:2
#SBATCH --requeue
#SBATCH --mem=25G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ig2283@nyu.edu

singularity exec --bind /scratch --nv --overlay /scratch/ig2283/overlay-25GB-500K.ext3:ro /scratch/ig2283/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c "
source /ext3/env.sh
source nanotron/bin/activate
cd /scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-2
CUDA_DEVICE_MAX_CONNECTIONS=1 ENABLE_TIMERS=1 DEBUG_DL=0 torchrun --nproc_per_node=2 run_train_nanotron.py --config-file run_train_nanotron_config.yaml
"
