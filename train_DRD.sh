#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --tmp=256G
#SBATCH --time=24:00:00

module load python/3.6
mkdir -p $SLURM_TMPDIR/env/temp
mkdir -p $SLURM_TMPDIR/data

cp -r ~/Venv/temp/* $SLURM_TMPDIR/env/temp
cp -r  ~/projects/rpp-bengioy/caotians/data/diabetic-retinopathy-detection $SLURM_TMPDIR/data

tar -xzf $SLURM_TMPDIR/data/diabetic-retinopathy-detection/images_224.tar.gz -C $SLURM_TMPDIR/data/diabetic-retinopathy-detection

source $SLURM_TMPDIR/env/temp/bin/activate
ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
python setup/DRDTrain.py --root_path=workspace/datasets-$SLURM_JOBID --exp="DRDDense" --batch-size=128 --no-visualize --save --workers=8