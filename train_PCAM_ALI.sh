#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=256G
#SBATCH --time=48:00:00
#SBATCH -a 0-8

PARRAY1=(0.999, 0.99, 0.9, 0.5, 0.1, 0.01, 0.001, 0.0001)

for i in {0..8}
    do
        let task_id=$i
        # printf $task_id"\n" # comment in for testing

        if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
        then
            p1=${PARRAY1[$i]}

            module load python/3.6
            mkdir -p $SLURM_TMPDIR/env/temp
            mkdir -p $SLURM_TMPDIR/data

            cp -r ~/Venv/temp/* $SLURM_TMPDIR/env/temp
            cp -r  ~/projects/rpp-bengioy/caotians/data/pcam $SLURM_TMPDIR/data

            source $SLURM_TMPDIR/env/temp/bin/activate
            ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
            export DISABLE_TQDM="True"

            python setup/PCAMTrainAli.py --root_path=workspace/datasets-$SLURM_JOBID --exp=PCAMDense_beta2_$p1 --batch-size=64 --no-visualize --save --workers=8 --beta2=$p1
         fi
    done
