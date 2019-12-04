import numpy as np
import csv, os
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import global_vars as Global
from sklearn.manifold import TSNE
from datasets.NIH_Chest import NIHChestBinaryTrainSplit
import seaborn as sns
import argparse
import os
import models as Models
from easydict import EasyDict
import _pickle
from datasets.NIH_Chest import NIHChest

All_OD1 = [
        'UniformNoise',
        #'NormalNoise',
        'MNIST',
        'FashionMNIST',
        'NotMNIST',
        #'CIFAR100',
        'CIFAR10',
        'STL10',
        'TinyImagenet',
        'MURAHAND',
        #'MURAWRIST',
        #'MURAELBOW',
        'MURAFINGER',
        #'MURAFOREARM',
        #'MURAHUMERUS',
        #'MURASHOULDER',
    ]
ALL_OD2 = [
            "PADChestAP",
            "PADChestL",
           "PADChestAPHorizontal",
           "PADChestPED"
]
d3_tags = ['Cardiomegaly', 'Pneumothorax', 'Nodule', 'Mass']

def proc_data(args, model, D1, d2s, tags):
    Out_X = []
    Out_Y = []
    Cat2Y = {}
    for y, D2 in enumerate(d2s):
        Cat2Y[tags[y]] = y + 1
        loader = DataLoader(D2, shuffle=True, batch_size=args.points_per_d2)
        for i, (X, _) in enumerate(loader):
            x = X.numpy()
            Out_X.append(x)
            Out_Y.append(np.ones(x.shape[0]) * (y + 1))
            break

    Out_X = np.concatenate(Out_X, axis=0)
    Out_Y = np.concatenate(Out_Y, axis=0)
    N_out = Out_X.shape[0]
    print(N_out)
    N_in = max(int(N_out * 0.2), args.points_per_d2)
    In_X = []
    for i in range(N_in):
        In_X.append(D1[i][0].numpy())
    In_Y = np.zeros(N_in)
    print(N_in, len(In_X), len(In_Y))
    Cat2Y["In-Data"] = 0
    ALL_X = np.concatenate((In_X, Out_X))
    ALL_Y = np.concatenate((In_Y, Out_Y))

    new_dataset = TensorDataset(torch.tensor(ALL_X))
    loader = DataLoader(new_dataset, batch_size=64)
    ALL_EMBS = []
    for i, (X,) in enumerate(loader):
        x = model.encode(X.cuda()).data.cpu().numpy()
        ALL_EMBS.append(x)
    ALL_EMBS = np.concatenate(ALL_EMBS, axis=0)

    return ALL_EMBS, ALL_Y, Cat2Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help="unique path for symlink to dataset")
    parser.add_argument('--seed', type=int, default=42, help='Random seed. (default 42)')
    parser.add_argument('--exp', '--experiment_id', type=str, default='test', help='The Experiment ID. (default test)')
    parser.add_argument('--embedding_function', type=str, default="VAE")
    parser.add_argument('--model_path', type=str, default="model_ref/Generic_VAE.HClass/NIHCC.dataset/BCE.max.512.d.12.nH.1024/model.best.pth")
    parser.add_argument('--points_per_d2', type=int, default=64)
    parser.add_argument('--lr', type=float, default=200)
    parser.add_argument('--perplexity', type=float, default=50.0)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--plot_percent', default=1.0, type=float)
    args = parser.parse_args()
    args.experiment_id = args.exp

    exp_data = []
    workspace_path = os.path.abspath('workspace')

    exp_list = args.experiment_id.split(',')
    exp_paths = []
    for exp_id in exp_list:
        experiments_path = os.path.join(workspace_path, 'experiments', exp_id)
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)

        # Make the experiment subfolders.
        for folder_name in exp_data:
            if not os.path.exists(os.path.join(experiments_path, folder_name)):
                os.makedirs(os.path.join(experiments_path, folder_name))
        exp_paths.append(experiments_path)

    if len(exp_list) == 1:
        args.experiment_path = exp_paths[0]
    else:
        print('Operating in multi experiment mode.', 'red')
        args.experiment_path = exp_paths

    #####################################################################################################
    if not args.load or not os.path.exists(os.path.join(args.experiment_path, "all_embs_UC1.npy")):

        D164 = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, 'NIHCC'), downsample=64)
        D1 = D164.get_D1_train()

        emb = args.embedding_function.lower()
        assert emb in ["vae", "ae", "ali"]
        dummy_args = EasyDict()
        dummy_args.exp = "foo"
        dummy_args.experiment_path = args.experiment_path
        if emb == "vae":
            model = Global.dataset_reference_vaes["NIHCC"][0]()
            home_path = Models.get_ref_model_path(dummy_args, model.__class__.__name__, D164.name,
                                                  suffix_str="BCE." + model.netid)
            model_path = os.path.join(home_path, 'model.best.pth')
        elif emb == "ae":
            model = Global.dataset_reference_autoencoders["NIHCC"][0]()

            home_path = Models.get_ref_model_path(dummy_args, model.__class__.__name__, D164.name,
                                                  suffix_str="MSE." + model.netid)
            model_path = os.path.join(home_path, 'model.best.pth')
        else:
            model = Global.dataset_reference_ALI["NIHCC"][0]()
            home_path = Models.get_ref_model_path(dummy_args, model.__class__.__name__, D164.name,
                                                  suffix_str="BCE." + model.netid)
            model_path = os.path.join(home_path, 'model.best.pth')

        model.load_state_dict(torch.load(model_path))

        model = model.to("cuda")

        d2s = []
        for y, d2 in enumerate(All_OD1):
            dataset = Global.all_datasets[d2]
            if 'dataset_path' in dataset.__dict__:
                print(os.path.join(args.root_path, dataset.dataset_path))
                D2 = dataset(root_path=os.path.join(args.root_path, dataset.dataset_path)).get_D2_test(D164)

            else:
                D2 = dataset().get_D2_test(D164)
            d2s.append(D2)

        ALL_EMBS, ALL_Y, Cat2Y = proc_data(args, model, D1, d2s, All_OD1)

        with open(os.path.join(args.experiment_path, "cat2y_UC1.pkl"), "wb") as fp:
            _pickle.dump(Cat2Y, fp)
        np.save(os.path.join(args.experiment_path, "all_y_UC1.npy"), ALL_Y)
        np.save(os.path.join(args.experiment_path, "all_embs_UC1.npy"), ALL_EMBS)

        #######################################################################################
        d2s = []
        for y, d2 in enumerate(ALL_OD2):
            dataset = Global.all_datasets[d2]
            if 'dataset_path' in dataset.__dict__:
                print(os.path.join(args.root_path, dataset.dataset_path))
                D2 = dataset(root_path=os.path.join(args.root_path, dataset.dataset_path)).get_D2_test(D164)

            else:
                D2 = dataset().get_D2_test(D164)
            d2s.append(D2)

        ALL_EMBS, ALL_Y, Cat2Y = proc_data(args, model, D1, d2s, ALL_OD2)

        with open(os.path.join(args.experiment_path, "cat2y_UC2.pkl"), "wb") as fp:
            _pickle.dump(Cat2Y, fp)
        np.save(os.path.join(args.experiment_path, "all_y_UC2.npy"), ALL_Y)
        np.save(os.path.join(args.experiment_path, "all_embs_UC2.npy"), ALL_EMBS)

        #########################################################################################
        d2s = []
        for d2 in d3_tags:
            D2 = NIHChest(root_path=os.path.join(args.root_path, 'NIHCC'), binary=True, test_length=5000,
                          keep_in_classes=[d2, ]).get_D2_test(D164)
            d2s.append(D2)
        ALL_EMBS, ALL_Y, Cat2Y = proc_data(args, model, D1, d2s, d3_tags)

        with open(os.path.join(args.experiment_path, "cat2y_UC3.pkl"), "wb") as fp:
            _pickle.dump(Cat2Y, fp)
        np.save(os.path.join(args.experiment_path, "all_y_UC3.npy"), ALL_Y)
        np.save(os.path.join(args.experiment_path, "all_embs_UC3.npy"), ALL_EMBS)

    else:
        pass

    for i in range(3):
        uc_tag = i+1
        ALL_EMBS = np.load(os.path.join(args.experiment_path, "all_embs_UC%i.npy"%uc_tag))
        with open(os.path.join(args.experiment_path, "cat2y_UC%i.pkl"%uc_tag), "rb") as fp:
            Cat2Y = _pickle.load(fp)
        ALL_Y = np.load(os.path.join(args.experiment_path, "all_y_UC%i.npy"%uc_tag))
        N=ALL_EMBS.shape[0]
        ALL_EMBS = ALL_EMBS.reshape(N, -1)
        N_plot = int(args.plot_percent * ALL_EMBS.shape[0])
        rand_inds = np.arange(ALL_EMBS.shape[0])
        np.random.shuffle(rand_inds)
        rand_inds = rand_inds[:N_plot]
        ALL_Y = ALL_Y[rand_inds]

        tsne = TSNE(perplexity=args.perplexity, learning_rate=args.lr, n_iter= args.n_iter)
        palette = sns.color_palette("bright", 10)
        from matplotlib.colors import ListedColormap
        my_cmap = ListedColormap(palette.as_hex())

        X_embedded = tsne.fit_transform(ALL_EMBS)

        X_embedded = X_embedded[rand_inds]
        fig, ax = plt.subplots()
        for k, cla in Cat2Y.items():
            target_inds = np.nonzero(ALL_Y == cla)
            ax.scatter(X_embedded[target_inds, 0].squeeze(), X_embedded[target_inds, 1].squeeze(), c=palette.as_hex()[cla], label=k)

        #ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=ALL_Y, cmap=my_cmap)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax.legend([k for k in Cat2Y.keys()])
        #plt.show()
        plt.savefig(os.path.join(args.experiment_path, "UC_%i_tsne.png"%uc_tag), dpi=100)
        #sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=ALL_Y, legend='full', palette=palette)
        print("done")

