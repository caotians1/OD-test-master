#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -n 1
source /pkgs/anaconda3/bin/activate pytorch

export DISABLE_TQDM="True"
#python /h/jeanlancel/OD-test-master/tsne_encoder.py --root_path="workspace/datasets" --exp="vaebce_padchest_emb" --points_per_d2=2048 --embedding_function="VAE" --dataset="PADChest" --encoder_loss="BCE" --perplexity=100 --lr=0.1 --n_iter=2000 --plot_percent=0.2 --umap
python /h/jeanlancel/OD-test-master/tsne_encoder.py --root_path="workspace/datasets" --exp="aebce_NIHCC" --points_per_d2=2048 --embedding_function="AE" --dataset="NIHCC" --encoder_loss="bce" --perplexity=100 --lr=0.1 --n_iter=2000 --plot_percent=0.2 --umap
python /h/jeanlancel/OD-test-master/tsne_encoder.py --root_path="workspace/datasets" --exp="aemse_NIHCC" --points_per_d2=2048 --embedding_function="AE" --dataset="NIHCC" --encoder_loss="MSE" --perplexity=100 --lr=0.1 --n_iter=2000 --plot_percent=0.2 --umap
python /h/jeanlancel/OD-test-master/tsne_encoder.py --root_path="workspace/datasets" --exp="aebce_padchest" --points_per_d2=2048 --embedding_function="AE" --dataset="PADChest" --encoder_loss="bce" --perplexity=100 --lr=0.1 --n_iter=2000 --plot_percent=0.2 --umap
python /h/jeanlancel/OD-test-master/tsne_encoder.py --root_path="workspace/datasets" --exp="aemse_padchest" --points_per_d2=2048 --embedding_function="AE" --dataset="PADChest" --encoder_loss="MSE" --perplexity=100 --lr=0.1 --n_iter=2000 --plot_percent=0.2 --umap
