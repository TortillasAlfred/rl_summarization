#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-adurand
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
#SBATCH --job-name=resubmission
#SBATCH --mem=1G

sleep $1
sbatch $2






