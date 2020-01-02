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

palette = sns.color_palette("bright", 10)
from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(palette.as_hex())

def plot_tsne(exp_path, uc, ax):
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
    if any([X_embedded, ALL_Y, Cat2Y] is None):
        raise FileNotFoundError("Missing some files from exp dir: %s" % exp_path)
    else:
        Y2Cat = dict([(v, k) for k,v in Cat2Y.items()])
        cat_labels = [Y2Cat[y] for y in ALL_Y]
        df = pd.DataFrame({'x':X_embedded[:,0], 'y':X_embedded[:,1], 'Dataset':cat_labels})
        sns.scatterplot('x', 'y', hue="Dataset", s=10.0, data=df, ax=ax)
        ax.set_axis_off()
        ax.legend(loc='upper right')
    return ax


def plot_result(exp_path, d1, uc, ax, order=None, c_order=None, with_x_tick=False):
    csv_data = np.load(os.path.join(exp_path, "data_UC%d_%s.npy" % (uc, d1)))
    assert os.path.isfile(os.path.join(exp_path, "headers_UC%d_%s.pkl" % (uc, d1)))
    with open(os.path.join(exp_path, "headers_UC%d_%s.pkl" % (uc, d1)), "rb") as fp:
        csv_headers = _pickle.load(fp)
    method_handles = csv_headers[0]
    weights = csv_data[0]
    uc_acc = csv_data[1]
    uc_roc = csv_data[2]
    uc_prc = csv_data[3]

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
            if m[n] + v[n] > 1.0:
                upper.append(1.0 - m[n])
            else:
                upper.append(v[n])
        return np.array([lower, upper])


    pp_accv = proc_var(sorted_accm, sorted_accv)
    pp_rocv = proc_var(sorted_rocm, sorted_rocv)
    pp_prcv = proc_var(sorted_prcm, sorted_prcv)

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
        ax.set_xticklabels(sorted_method_handles, rotation=45, ha='right')
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_yticks(np.linspace(0,1,5))
    ax.set_ylim(0, 1.1)
    #ax.legend()
    return full_inds, c_order


if __name__ == "__main__":
    fig, ax = plt.subplots(3,1)
    inds, corder = plot_result("workspace/experiments/nih_res", "NIHCC", 1, ax[0])
    plot_result("workspace/experiments/nih_res", "NIHCC", 2, ax[1], inds, c_order=corder)
    plot_result("workspace/experiments/nih_res", "NIHCC", 3, ax[2], inds, c_order=corder, with_x_tick=True)
    plt.subplots_adjust(hspace=0.1)
    fig.show()
    print('done')
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--square_col_0_path', type=str)
    #parser.add_argument('--square_col_1_path', type=str)
    #parser.add_argument('--perf_path', type=str)
    #parser.add_argument('--save_path', type=str, default='test', help='The save path. (default test)')

    #args = parser.parse_args()

