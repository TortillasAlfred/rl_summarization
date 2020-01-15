#!/bin/bash
#SBATCH --account=def-lulam40                            # Account with resources
#SBATCH --cpus-per-task=6                                # Number of CPUs
#SBATCH --mem=35G                                        # memory (per node)
#SBATCH --time=0-03:00                                   # time (DD-HH:MM)
#SBATCH --mail-user=mathieu.godbout.3@ulaval.ca          # Where to email
#SBATCH --mail-type=FAIL                                 # Email when a job fails

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

date
SECONDS=0

# The $@ transfers all args passed to this bash file to the Python script
# i.e. a call to 'sbatch $sbatch_args this_launcher.sh --arg1=0 --arg2=True'
# will call 'python my_script.py --arg1=0 --arg2=True'
python src/domain/analysis.py

# Utility to show job duration in output file
diff=$SECONDS
echo "$(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
date
