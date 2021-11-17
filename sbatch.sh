#!/bin/bash
# Usage: sbatch run_slurm.sh

SBATCH --partition=macondo --qos=normal
SBATCH --nodelist=macondo-dgx1
SBATCH --time=48:00:00
SBATCH --nodes=1
SBATCH --ntasks-per-node=1
SBATCH --cpus-per-task=20
SBATCH --mem=20G
SBATCH --gres=gpu:1
SBATCH --job-name="brave_test"
SBATCH --output=logs/brave_test_slurm_%A.out
SBATCH --error=logs/brave_test_slurm_%A.err

######################
# Begin work section #
######################
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

##########################################
# Setting up virtualenv / conda / docker #
##########################################
# example here if using virtualenv
source /vision/u/naagnes/github/brave-pytorch/.venv/bin/activate
echo "Virtual Env Activated"

##############################################################
# Setting up LD_LIBRARY_PATH or other env variable if needed #
##############################################################
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:/usr/lib/x86_64-linux-gnu 
echo "Working with the LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

###################
# Run your script #
###################
cd /vision/u/naagnes/github/brave-pytorch/
echo "running command : python train.py"
python train.py --accelerator gpu --gpus 1 --max_steps 10 --batch_size 4
