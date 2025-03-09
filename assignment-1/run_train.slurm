#!/bin/sh
#SBATCH --job-name=job_wgpu
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --mem=25G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ig2283@nyu.edu

singularity exec --bind /scratch --nv --overlay /scratch/ig2283/overlay-25GB-500K.ext3:ro /scratch/ig2283/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash -c "
source /ext3/env.sh
conda activate bdml_env
cd /scratch/ig2283/Workspace/nyu-big-data-and-ml/assignment-1
python train.py
"
