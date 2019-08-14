#!/bin/sh
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --tmp=256G
#SBATCH --time=3:00:00
ssh -N -R 9611:localhost:9611 beluga2 &
module load python/3.6

mkdir -p $SLURM_TMPDIR/env/temp
mkdir -p $SLURM_TMPDIR/data

cp -r ~/Venv/temp/* $SLURM_TMPDIR/env/temp
cp -r  ~/projects/rpp-bengioy/caotians/data/* $SLURM_TMPDIR/data

mkdir -p $SLURM_TMPDIR/data/NIHCC/images_224

tar -xzf $SLURM_TMPDIR/data/NIHCC/images_224.tar.gz -C $SLURM_TMPDIR/data/NIHCC/images_224 --strip-components=1
tar -xzf $SLURM_TMPDIR/data/MURA/images_224.tar.gz -C $SLURM_TMPDIR/data/MURA
tar -xf $SLURM_TMPDIR/data/PADChest/images-64.tar -C $SLURM_TMPDIR/data/PADChest
tar -xzf $SLURM_TMPDIR/data/diabetic-retinopathy-detection/images_224.tar.gz -C $SLURM_TMPDIR/data/diabetic-retinopathy-detection
tar -xzf $SLURM_TMPDIR/data/RIGA-dataset/images_224.tar.gz -C $SLURM_TMPDIR/data/RIGA-dataset
unzip $SLURM_TMPDIR/data/IDC/IDC_regular_ps50_idx5.zip -d $SLURM_TMPDIR/data/IDC/images

source $SLURM_TMPDIR/env/temp/bin/activate
python setup_datasets.py
ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID

mkdir -p $SLURM_TMPDIR/jupyter
export JUPYTER_RUNTIME_DIR=$SLURM_TMPDIR/jupyter
jupyter notebook --port=9611 --no-browser
