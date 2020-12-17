#!/bin/bash

#SBATCH --account=def-lulam50                                                  # Account with resources
#SBATCH --cpus-per-task=8                                                      # Number of CPUs
#SBATCH --mem=60G                                                              # memory (per node)
#SBATCH --time=0-03:00                                                         # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca                                # Where to email
#SBATCH --mail-type=FAIL                                                       # Email when a job fails
#SBATCH --output=/project/def-lulam50/magod/rl_summ/slurm_outputs/%A.out       # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --signal=SIGUSR1@90                                                    # Killing signal 90 seconds before job end

mkdir /project/def-lulam50/magod/rl_summ/slurm_outputs/

mkdir $SLURM_TMPDIR/finished_files
cp /scratch/magod/summarization_datasets/cnn_dailymail/tarred/finished_files/*.tar $SLURM_TMPDIR/finished_files/

source ~/venvs/default/bin/activate

python -um src.scripts.ngrams_calc --data_path $SLURM_TMPDIR/
