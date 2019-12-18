#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=256G
#SBATCH --time=48:00:00
#SBATCH -a 0-6

PARRAY1=(0 1 2 3 4 5 6 7 8 9)

for i in {0..10}
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
            cp -r  ~/projects/rpp-bengioy/caotians/data/* $SLURM_TMPDIR/data

            mkdir -p $SLURM_TMPDIR/data/NIHCC/images_224
            tar -xzf $SLURM_TMPDIR/data/NIHCC/images_224.tar.gz -C $SLURM_TMPDIR/data/NIHCC/images_224 --strip-components=1
            tar -xzf $SLURM_TMPDIR/data/MURA/images_224.tar.gz -C $SLURM_TMPDIR/data/MURA
            tar -xf $SLURM_TMPDIR/data/PADChest/images-64.tar -C $SLURM_TMPDIR/data/PADChest

            source $SLURM_TMPDIR/env/temp/bin/activate
            python setup_datasets.py
            ln -sf $SLURM_TMPDIR/data workspace/datasets-$SLURM_JOBID
            export DISABLE_TQDM="True"
            python Chest_eval_rand_seeds.py --root_path=workspace/datasets-$SLURM_JOBID --exp=Chest_eval_new_pad_seed_$p1 --seed=$p1 --batch-size=64 --no-visualize --save --workers
         fi
    done
