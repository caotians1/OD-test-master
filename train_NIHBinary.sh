#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --time=10:00:00
#module load python/3.6
cp -r ~/Venv/temp $SLURM_TMPDIR/env/temp
cp -r  ~/projects/rpp-bengioy/caotians SLURM_TMPDIR/data
cp -r ~/OD-test-master SLURM_TMPDIR/OD-test-master
source $SLURM_TMPDIR/env/temp/activate
python setup/NIHTrain_binary.py