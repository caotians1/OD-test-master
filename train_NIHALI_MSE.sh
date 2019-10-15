#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --tmp=256G
#SBATCH --time=30:00:00

module load python/3.6
mkdir -p $SLURM_TMPDIR/env/temp
mkdir -p $SLURM_TMPDIR/data

cp -r ~/Venv/temp/* $SLURM_TMPDIR/env/temp
cp -r  ~/projects/rpp-bengioy/caotians/data/* $SLURM_TMPDIR/data

mkdir -p $SLURM_TMPDIR/data/NIHCC/images_224
tar -xzf $SLURM_TMPDIR/data/NIHCC/images_224.tar.gz -C $SLURM_TMPDIR/data/NIHCC/images_224 --strip-components=1
tar -xzf $SLURM_TMPDIR/data/MURA/images_224.tar.gz -C $SLURM_TMPDIR/data/MURA
tar -xf $SLURM_TMPDIR/data/PADChest/images-64.tar -C $SLURM_TMPDIR/data/PADChest

source $SLURM_TMPDIR/env/temp/bin/activate
python setup_datasets.py
ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
python setup/NIHTrainALILikeResAE.py --root_path=workspace/datasets-$SLURM_JOBID --exp="NIHAliLikeAE" --batch-size=64 --no-visualize --save --workers=8