#!/bin/bash                                      
#SBATCH --account=rpp-corbeilj                             # Account with resources
#SBATCH --cpus-per-task=16                                     # Number of CPUs
#SBATCH --mem=10G                                              # memory (per node)
#SBATCH --time=0-12:00                                        # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca               # Where to email
#SBATCH --mail-type=FAIL                                      # Email when a job fails
#SBATCH --output=/scratch/magod/rouge_npy/%A_%a.out          # Default write output on scratch, to jobID_arrayID.out file

source ~/venvs/default/bin/activate

date
SECONDS=0

# You can access the array ID via $SLURM_ARRAY_TASK_ID

# The $@ transfers all args passed to this bash file to the Python script
# i.e. a call to 'sbatch $sbatch_args this_launcher.sh --arg1=0 --arg2=True'
# will call 'python my_script.py --arg1=0 --arg2=True'
python -um src.scripts.reward_numpy --data_path=/scratch/magod/summarization_datasets/cnn_dailymail/data/ --dataset=train --vectors_cache=/scratch/magod/embeddings/ --target_dir=/scratch/magod/summarization_datasets/cnn_dailymail/data/rouge_npy/
# Utility to show job duration in output file
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date