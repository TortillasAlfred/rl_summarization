#!/bin/bash

#SBATCH --account=rrg-corbeilj-ac                                              # Account with resources
#SBATCH --cpus-per-task=30                                                     # Number of CPUs
#SBATCH --mem=100G                                                             # memory (per node)
#SBATCH --time=0-03:00                                                         # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca                                # Where to email
#SBATCH --mail-type=FAIL                                                       # Email when a job fails
#SBATCH --output=/project/def-lulam50/magod/rl_summ/slurm_outputs/%A.out       # Default write output on scratch, to jobID_arrayID.out file
#SBATCH --signal=SIGUSR1@90                                                    # Killing signal 90 seconds before job end

mkdir /project/def-lulam50/magod/rl_summ/slurm_outputs/

cp /scratch/magod/summarization_datasets/cnn_dailymail/tarred/linsit_dataset.tar $SLURM_TMPDIR/
tar -xf $SLURM_TMPDIR/linsit_dataset.tar -C $SLURM_TMPDIR/

source ~/venvs/default/bin/activate

python -um src.scripts.linsit_exp --data_path $SLURM_TMPDIR/linsit_dataset/
