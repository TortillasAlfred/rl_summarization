#!/bin/bash

#SBATCH --account=rrg-corbeilj-ac                                              # Account with resources
#SBATCH --cpus-per-task=4                                                      # Number of CPUs
#SBATCH --gres=gpu:t4:1                                                        # Number of GPUs (per node)
#SBATCH --mem=46G                                                              # memory (per node)
#SBATCH --time=3-00:00                                                         # time (DD-HH:MM)
#SBATCH --mail-user=jeremie.gauthier.1@ulaval.ca                               # Where to email
#SBATCH --mail-type=FAIL                                                       # Email when a job fails
#SBATCH --output=/project/def-adurand/jegauth/rl_summ/output/%A_%a.out         # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --array=0-119            					                           # Launch an array of 120 jobs
#SBATCH --signal=SIGUSR1@90                                                    # Killing signal 90 seconds before job end

echo "TRAIN RL SUMMARIZATION"
echo "### IMPORTANT: Create /output directory manually before running the process ###"

# Load the full dataset in $SLURM_TMPDIR/
cp /home/jegauth/main/data.zip $SLURM_TMPDIR/
cd $SLURM_TMPDIR/
unzip -qq data.zip

# Change directory
cd /home/jegauth/main/rl_summarization

# Activate virtual environnement
source /home/jegauth/main/DS_env/bin/activate

# Load the necessary modules
module load StdEnv/2020 gcc/9.3.0 arrow/2.0.0
module load python/3.8.2 scipy-stack

# Run the program
/home/jegauth/main/DS_env/bin/python3 -m src.scripts.training --data_path $SLURM_TMPDIR/data --default_save_path /project/def-adurand/jegauth/rl_summ/results --weights_save_path /project/def-adurand/jegauth/rl_summ/weights --job_index $SLURM_ARRAY_TASK_ID