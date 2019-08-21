#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --tmp=128G
#SBATCH --time=4:00:00

mkdir -p $SLURM_TMPDIR/data

cp -r  /lustre04/scratch/cohenjos/PC/images-299 $SLURM_TMPDIR/data
tar -I pigz -cf $SLURM_TMPDIR/data/images-299.tar.gz --directory=$SLURM_TMPDIR/data images-299

cp $SLURM_TMPDIR/data/images-299.tar.gz ~/projects/rpp-bengioy/caotians/data/PADChest/

#source $SLURM_TMPDIR/env/temp/bin/activate
#python setup_datasets.py
#ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
#python setup/NIHTrain_binary.py --root_path=workspace/datasets-$SLURM_JOBID --exp="nihbinary_test" --batch-size=64 --no-visualize --save --workers=8