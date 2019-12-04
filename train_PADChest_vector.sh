#!/bin/bash
#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -n 1

source /pkgs/anaconda3/bin/activate pytorch

#export DISABLE_TQDM="True"

python setup/PAD/PADTrain_binary.py --root_path=workspace/datasets --exp="PADChestDenseLateralView" --batch-size=64 --no-visualize --save --workers=8
