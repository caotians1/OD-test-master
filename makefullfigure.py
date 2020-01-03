"""
First run tsne_encoder.py until the visualizations look good, and then set tsne_cache path to that experiment dir.
"""


import numpy as np
import csv, os
import torch
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import os
import models as Models
from easydict import EasyDict
import _pickle
from process_results_multiple import weighted_std
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import skimage.io as skio


from matplotlib.colors import ListedColormap


def plot_tsne(exp_path, uc, ax, color_space=None, n_color_max=30, cat2color=None):
    X_embedded = None
    ALL_Y = None
    Cat2Y = None
    for (dirpath, dirnames, filenames) in os.walk(exp_path):
        for filename in filenames:
            if ('embedded_UC%i' % uc) in filename:
                X_embedded = np.load(os.path.join(exp_path, filename))
            elif ('selectedY_UC%i' % uc) in filename:
                ALL_Y = np.load(os.path.join(exp_path, filename))
            elif ('cat2y_UC%i' % uc) in filename:
                with open(os.path.join(exp_path,filename), "rb") as fp:
                    Cat2Y = _pickle.load(fp)
    if any([X_embedded is None, ALL_Y is None, Cat2Y is None]):
        raise FileNotFoundError("Missing some files from exp dir: %s" % exp_path)
    else:
        Y2Cat = dict([(v, k) for k,v in Cat2Y.items()])
        cat_labels = [Y2Cat[y] for y in ALL_Y]

        if color_space is None:
            assert cat2color is None
            color_space = sns.color_palette("hls", 20)
            cat2color = {}
            i = 0
            for y, cat in Y2Cat.items():
                cat2color[cat] = color_space[i]
                i += 1
            this_color_list = [c for cat, c in cat2color.items()]
            this_cmap = ListedColormap(this_color_list)
        else:
            assert  cat2color is not None
            n_used_colors = len(list(cat2color.values()))
            this_color_list = []
            for y, cat in Y2Cat.items():
                if cat in cat2color.keys():
                    this_color_list.append(cat2color[cat])
                else:
                    # new category
                    this_color_list.append(color_space[n_used_colors])
                    cat2color[cat] = color_space[n_used_colors]
                    n_used_colors += 1
            this_cmap = ListedColormap(this_color_list)

        df = pd.DataFrame({'x':X_embedded[:,0], 'y':X_embedded[:,1], 'Dataset':cat_labels})
        sns.scatterplot('x', 'y', hue="Dataset", s=1.0, data=df, ax=ax, linewidth=0, palette=cat2color)
        ax.set_axis_off()
        ax.legend_.remove()
        #ax.legend(loc='upper right')
    return ax, color_space, cat2color

def plot_image(file_path, ax):
    img = skio.imread(file_path)
    if len(img.shape)>2:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    return ax

def plot_result(exp_path, d1, uc, ax, order=None, c_order=None, with_x_tick=False, keep_only_handles=None, alias=None):
    csv_data = np.load(os.path.join(exp_path, "data_UC%d_%s.npy" % (uc, d1)))
    assert os.path.isfile(os.path.join(exp_path, "headers_UC%d_%s.pkl" % (uc, d1)))
    with open(os.path.join(exp_path, "headers_UC%d_%s.pkl" % (uc, d1)), "rb") as fp:
        csv_headers = _pickle.load(fp)
    method_handles = csv_headers[0]
    weights = csv_data[0]
    uc_acc = csv_data[1]*100
    uc_roc = csv_data[2]*100
    uc_prc = csv_data[3]*100

    accm = np.average(uc_acc, axis=(1, 2), weights=weights)
    rocm = np.average(uc_roc, axis=(1, 2), weights=weights)
    prcm = np.average(uc_prc, axis=(1, 2), weights=weights)
    accv = weighted_std(uc_acc, weights, axis=(1, 2))
    rocv = weighted_std(uc_roc, weights, axis=(1, 2))
    prcv = weighted_std(uc_prc, weights, axis=(1, 2))

    if order is None:
        group1_handles_inds = []
        group2_handles_inds = []
        for i, handle in enumerate(method_handles):
            if ("ae" in handle.lower()) or ("ali" in handle.lower()):
                group2_handles_inds.append(i)
            else:
                group1_handles_inds.append(i)
        group1_sum = accm[group1_handles_inds] + rocm[group1_handles_inds] + prcm[group1_handles_inds]
        group2_sum = accm[group2_handles_inds] + rocm[group2_handles_inds] + prcm[group2_handles_inds]

        sorted_inds_g1 = [group1_handles_inds[i] for i in np.argsort(group1_sum)]
        sorted_inds_g2 = [group2_handles_inds[i] for i in np.argsort(group2_sum)]
        assert type(sorted_inds_g1) is list
        full_inds = np.array(sorted_inds_g1 + sorted_inds_g2)
    else:
        full_inds = order
    sorted_accm = accm[full_inds]
    sorted_accv = accv[full_inds]

    sorted_rocm = rocm[full_inds]
    sorted_rocv = rocv[full_inds]

    sorted_prcm = prcm[full_inds]
    sorted_prcv = prcv[full_inds]

    sorted_method_handles = [method_handles[i] for i in full_inds]

    def proc_var(m, v):
        upper = []
        lower = []
        for n in range(m.shape[0]):
            if m[n] - v[n] < 0.0:
                lower.append(m[n])
            else:
                lower.append(v[n])
            if m[n] + v[n] > 100.0:
                upper.append(100.0 - m[n])
            else:
                upper.append(v[n])
        return np.array([lower, upper])


    pp_accv = proc_var(sorted_accm, sorted_accv)
    pp_rocv = proc_var(sorted_rocm, sorted_rocv)
    pp_prcv = proc_var(sorted_prcm, sorted_prcv)

    if keep_only_handles is not None:
        keep_inds = []
        for i, method in enumerate(sorted_method_handles):
            if method in keep_only_handles:
                keep_inds.append(i)
            else:
                print("leaving out %s" % method)
        keep_inds = np.array(keep_inds)
        sorted_accm = sorted_accm[keep_inds]
        sorted_rocm = sorted_rocm[keep_inds]
        sorted_prcm = sorted_prcm[keep_inds]
        pp_accv = pp_accv[:,keep_inds]
        pp_rocv = pp_rocv[:,keep_inds]
        pp_prcv = pp_prcv[:,keep_inds]
        sorted_method_handles = [sorted_method_handles[i] for i in keep_inds]
        full_inds = full_inds[keep_inds]

    ind = np.arange(len(sorted_accm))  # the x locations for the groups
    width = 0.25  # the width of the bars

    rb = cm.get_cmap('rainbow')
    grad = np.linspace(0, 1, len(sorted_accm))
    colors = [rb(g) for g in grad]

    if c_order is None:
        this_order = np.arange(0,len(full_inds))
        c_order = {}
        for c, i in zip(this_order, full_inds):
            c_order[i]=c
    else:
        this_order = [c_order[i] for i in full_inds]

    sorted_colors = [colors[i] for i in this_order]
    sorted_colors_0 = [(color[0], color[1], color[2], 0.3,) for color in sorted_colors]
    sorted_colors_1 = [(color[0], color[1], color[2], 0.7,) for color in sorted_colors]



    rects1 = ax.bar(ind - width * 0.99, sorted_accm, width, yerr=pp_accv,
                    label='Accuracy', color=sorted_colors_1)
    rects2 = ax.bar(ind, sorted_rocm, width, yerr=pp_rocv,
                    label='AUROC', color=sorted_colors_0)
    rects3 = ax.bar(ind + width * 0.99, sorted_prcm, width, yerr=pp_prcv,
                    label='AUPRC',  color=sorted_colors)
    if with_x_tick:
        ax.set_xticks(ind)
        if alias is not None:
            this_method_handles = []
            for handle in sorted_method_handles:
                if handle in alias:
                    this_method_handles.append(alias[handle])
                else:
                    this_method_handles.append(handle)
        else:
            this_method_handles = sorted_method_handles
        ax.set_xticklabels(this_method_handles, rotation=-45, ha='left')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_yticks(np.linspace(0,100,3))
    ax.set_ylim(0, 110)
    #ax.legend()
    return ax, full_inds, c_order


if __name__ == "__main__":
    PREVIEW_DOUBLE=True
    #fig, ax = plt.subplots(3,2)
    # Plot figure with subplots of different sizes
    if PREVIEW_DOUBLE:
        fig = plt.figure(1, figsize=(6, 3.5), dpi=109*2)
    else:
        fig = plt.figure(1, figsize=(6, 3.5), dpi=109)       # dpi set to match preview to print size on a 14 inch tall, 1440p monitor.
    # set up subplot grid
    gridspec.GridSpec(3, 6)     # nrow by n col         # effectively 1 inch squres

    plt.rc('font', size=5)  # controls default text sizes
    plt.rc('axes', titlesize=5)  # fontsize of the axes title
    plt.rc('axes', labelsize=5)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=5)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the tick labels
    plt.rc('legend', fontsize=5)  # legend fontsize
    plt.rc('figure', titlesize=10)  # fontsize of the figure title
    #plt.rcParams.update({'font.size': 9})
    # large subplot
    ax_L0 = plt.subplot2grid((3, 6), (0, 2), colspan=4, rowspan=1)
    ax_L0.yaxis.tick_right()
    kiohandles = ["prob_threshold/0", "score_svm/0", "binclass/0", "odin/0", "Maha", "Maha1layer", "svknn",
                  "12Layer-AE-BCE", "12Layer-AE-MSE", "12Layer-VAE-BCE", "12Layer-VAE-MSE", "ALI_reconst/0",
                  "knn/1", "knn/8", "bceaeknn/8", "vaebceaeknn/8",
                  "mseaeknn/8", "vaemseaeknn/8", "bceaeknn/1", "vaebceaeknn/1",
                  "mseaeknn/1", "vaemseaeknn/1"
                  ]

    alias={"prob_threshold/0": "Prob. threshold", "score_svm/0": "Score SVM", "binclass/0": "Binary classifier",
           "odin/0":"ODIN", "Maha": "Mahalanobis", "Maha1layer":"Single layer Maha.", "svknn":"Feature knn",
           "12Layer-AE-BCE":"Reconst. AEBCE", "12Layer-AE-MSE":"Reconst. AEMSE", "12Layer-VAE-MSE":"Reconst. VAEMSE",
           "12Layer-VAE-BCE": "Reconst. VAEBCE", "ALI_reconst/0":"Reconst. ALI",
           "knn/1":"KNN-1", "knn/8":"KNN-8", "bceaeknn-8":"AEBCE-KNN-8", "vaebceaeknn-8":"VAEBCE-KNN-8",
          "mseaeknn/8":"AEMSE-KNN-8", "vaemseaeknn/8":"VAEMSE-KNN-8", "bceaeknn/1":"AEBCE-KNN-1", "vaebceaeknn/1":"VAEBCE-KNN-1",
          "mseaeknn/1":"AEMSE-KNN-1", "vaemseaeknn/1":"VAEMSE-KNN-1"
           }

    catalias={"UniformNoise":"Noise", "FashionMNIST":"Fashion", "PADChestAP":"Ant. Pos.", "PADChestL":"Lateral", "PADChestAPHorizontal": "AP Horizontal", "PADChestPED":"Pediatric"}

    _, inds, corder = plot_result("workspace/experiments/nih_res", "NIHCC", 1, ax_L0, keep_only_handles=kiohandles)
    ax_L1 = plt.subplot2grid((3, 6), (1, 2), colspan=4, rowspan=1)
    ax_L1.yaxis.tick_right()
    plot_result("workspace/experiments/nih_res", "NIHCC", 2, ax_L1, inds, c_order=corder, keep_only_handles=kiohandles)

    ax_L2 = plt.subplot2grid((3, 6), (2, 2), colspan=4, rowspan=1)
    ax_L2.yaxis.tick_right()
    plot_result("workspace/experiments/nih_res", "NIHCC", 3, ax_L2, inds, c_order=corder, with_x_tick=True,
                keep_only_handles=kiohandles, alias=alias)

    ax_s11 = plt.subplot2grid((3, 6), (0, 1), colspan=1, rowspan=1)
    _, color_space, cat2color = plot_tsne("workspace/experiments/ALI_NIHCC", 1, ax_s11)
    ax_s21 = plt.subplot2grid((3, 6), (1, 1), colspan=1, rowspan=1)
    plot_tsne("workspace/experiments/ALI_NIHCC", 2, ax_s21, color_space=color_space, cat2color=cat2color)
    ax_s31 = plt.subplot2grid((3, 6), (2, 1), colspan=1, rowspan=1)
    plot_tsne("workspace/experiments/ALI_NIHCC", 3, ax_s31, color_space=color_space, cat2color=cat2color)


    cats = list(cat2color.keys())
    for i, cat in enumerate(cats):
        if cat in catalias:
            cats[i] = catalias[cat]
    colors = list(cat2color.values())
    id = cats.index("In-Data")
    cats = ["In-data", ] + cats[:id] + cats[id+1:]
    colors = [colors[id], ] + colors[:id] + colors[id+1:]
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in colors]
    fig.legend(
               markers, cats,
               title="Scatter plot legend",
               numpoints=1, loc='upper left', bbox_to_anchor=(0.0, 0.2),
               ncol=3,
               markerscale=0.6,
               labelspacing=0.3,    # default 0.5
               columnspacing=1.0,   # default 2.0
               borderaxespad=0.5,   # default 0.5
               )


    ax_s10 = plt.subplot2grid((3, 6), (0, 0), colspan=1, rowspan=1)
    plot_image("sample_images/mura.png", ax_s10)
    ax_s20 = plt.subplot2grid((3, 6), (1, 0), colspan=1, rowspan=1)
    plot_image("sample_images/padchest_lateral.png", ax_s20)
    ax_s30 = plt.subplot2grid((3, 6), (2, 0), colspan=1, rowspan=1)
    plot_image("sample_images/pneumothorax.png", ax_s30)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.20, top=0.95, right=0.95, left=0.01)

    plt.savefig("NIHCC_sample.svg")
    plt.savefig("NIHCC_sample.png")
    fig.show()

    print('done')
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--square_col_0_path', type=str)
    #parser.add_argument('--square_col_1_path', type=str)
    #parser.add_argument('--perf_path', type=str)
    #parser.add_argument('--save_path', type=str, default='test', help='The save path. (default test)')

    #args = parser.parse_args()

