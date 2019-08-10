#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --tmp=128G
#SBATCH --time=4:00:00

module load python/3.6
source ~/Venv/temp/bin/activate
mkdir -p $SLURM_TMPDIR/data

cp -r  ~/scratch/RIGA-dataset $SLURM_TMPDIR/data

unzip $SLURM_TMPDIR/data/RIGA-dataset/MESSIDOR.zip -d $SLURM_TMPDIR/data/RIGA-dataset/
unzip $SLURM_TMPDIR/data/RIGA-dataset/Magrabia.zip -d $SLURM_TMPDIR/data/RIGA-dataset/
unzip $SLURM_TMPDIR/data/RIGA-dataset/BinRushedcorrected.zip -d $SLURM_TMPDIR/data/RIGA-dataset/
#unzip $SLURM_TMPDIR/data/diabetic-retinopathy-detection/trainLabels.csv

python preproc/prepare_RIGA.py --source_dir=$SLURM_TMPDIR/data/RIGA-dataset --image_dir=train --proc_dir=$SLURM_TMPDIR/data/RIGA-dataset/images_224

tar -I pigz -cf $SLURM_TMPDIR/data/RIGA-dataset/images_224.tar.gz --directory=$SLURM_TMPDIR/data/RIGA-dataset images_224
mkdir ~/projects/rpp-bengioy/caotians/data/RIGA-dataset
cp $SLURM_TMPDIR/data/RIGA-dataset/images_224.tar.gz ~/projects/rpp-bengioy/caotians/data/RIGA-dataset/


#source $SLURM_TMPDIR/env/temp/bin/activate
#python setup_datasets.py
#ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
#python setup/NIHTrain_binary.py --root_path=workspace/datasets-$SLURM_JOBID --exp="nihbinary_test" --batch-size=64 --no-visualize --save --workers=8