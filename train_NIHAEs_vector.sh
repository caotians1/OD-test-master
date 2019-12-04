#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH -a 0-4

PARRAY1=(NIHTrainAE.py NIHTrainAEBCE.py NIHTrainVAE.py NIHTrainVAEBCE.py NIHTrainALILikeAE.py NIHTrainALILikeAEBCE.py PADTrainALILikeVAE.py PADTrainALILikeVAEBCE.py)

for i in {0..8}
    do
        let task_id=$i
        # printf $task_id"\n" # comment in for testing

        if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
        then
            p1=${PARRAY1[$i]}
            source /pkgs/anaconda3/bin/activate pytorch
            #module load python/3.6
            #mkdir -p $SLURM_TMPDIR/env/temp
            #mkdir -p $SLURM_TMPDIR/data

            #cp -r ~/Venv/temp/* $SLURM_TMPDIR/env/temp
            #cp -r  ~/projects/rpp-bengioy/caotians/data/NIHCC $SLURM_TMPDIR/data

            #mkdir -p $SLURM_TMPDIR/data/NIHCC/images_224
            #tar -xzf $SLURM_TMPDIR/data/NIHCC/images_224.tar.gz -C $SLURM_TMPDIR/data/NIHCC/images_224 --strip-components=1

            #source $SLURM_TMPDIR/env/temp/bin/activate
            #ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
            export DISABLE_TQDM="True"

            python setup/NIH/$p1 --root_path=workspace/datasets --exp=model_ref_nih --batch-size=64 --no-visualize --save --workers=8
         fi
    done
