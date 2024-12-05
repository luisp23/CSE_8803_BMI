#!/bin/bash
#SBATCH -J TrainContinualDiffuserCheetahVel          # Job name
#SBATCH -N 1                             # Number of nodes required
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --gres=gpu:A40:1                # Request one H100 GPU
#SBATCH --mem=32G                        # Adjust memory allocation 
#SBATCH --time=16:00:00                  # Time limit (16 hours)
#SBATCH -o Report-%j.out                 # Standard output and error log
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail notifications for job events
#SBATCH --mail-user=lpimentel3@gatech.edu # Email address for notifications



# Activate your Python environment (if needed)
source activate continual_diffuser    # Replace with your conda environment name

# Change to the directory where the job was submitted from
cd /home/hice1/lpimentel3/scratch/continual_diffuser/CuGRO

export CUDA_VISIBLE_DEVICES=0 
srun python eval_continual_diffuser.py \
    --env "meta_world" \
    --data_mode "continual_diffuser" \
    --trajectory_horizon 32 \
    --ending_task 3