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
cp -r  ~/projects/rpp-bengioy/caotians/data/pcam $SLURM_TMPDIR/data

#tar -xzf $SLURM_TMPDIR/data/diabetic-retinopathy-detection/images_224.tar.gz -C $SLURM_TMPDIR/data/diabetic-retinopathy-detection

source $SLURM_TMPDIR/env/temp/bin/activate
ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
export DISABLE_TQDM="True"
python setup/DRDTrain2.py --root_path=workspace/datasets-$SLURM_JOBID --exp="PCAMDense" --batch-size=64 --no-visualize --save --workers=8