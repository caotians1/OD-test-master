#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=256G
#SBATCH --time=48:00:00
#SBATCH -a 0-8

PARRAY1=(DRDTrainAEBCE.py DRDTrainAEMSE.py DRDTrainVAEBCE.py DRDTrainVAEMSE.py DRDTrainALILikeBCE.py DRDTrainALILikeMSE.py DRDTrainALILikeVAEBCE.py DRDTrainALILikeVAEMSE.py)
PARRAY2=(DRDAEBCE DRDAEMSE DRDVAEBCE DRDVAEMSE DRDALIBCE DRDALIMSE DRDALIVAEBCE DRDALIVAEMSE)

for i in {0..8}
    do
        let task_id=$i
        # printf $task_id"\n" # comment in for testing

        if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
        then
            p1=${PARRAY1[$i]}
            p2=${PARRAY2[$i]}
            module load python/3.6
            mkdir -p $SLURM_TMPDIR/env/temp
            mkdir -p $SLURM_TMPDIR/data

            cp -r ~/Venv/temp/* $SLURM_TMPDIR/env/temp
            cp -r  ~/projects/rpp-bengioy/caotians/data/diabetic-retinopathy-detection $SLURM_TMPDIR/data

            tar -xzf $SLURM_TMPDIR/data/diabetic-retinopathy-detection/images_224.tar.gz -C $SLURM_TMPDIR/data/diabetic-retinopathy-detection

            source $SLURM_TMPDIR/env/temp/bin/activate
            ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
            export DISABLE_TQDM="True"

            python setup/DRD/$p1 --root_path=workspace/datasets-$SLURM_JOBID --exp=$p2 --batch-size=64 --no-visualize --save --workers=8
         fi
    done
