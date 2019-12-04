import numpy as np
import csv, os
import torch
import matplotlib
import matplotlib.pyplot as plt


def weighted_std(values, weights, axis=None):
    if axis is None:
        axis = np.arange(len(values.shape))

    average = np.average(values, weights=weights, axis=axis)
    for axi in axis:
        average = np.expand_dims(average, axi)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)

def make_plot(filename, d2_handles, title, results):
    rec_error = {
        "reconst_thresh/0": "12Layer-AE-BCE",
        "reconst_thresh/1": "12Layer-AE-MSE",
        "reconst_thresh/2": "12Layer-VAE-BCE",
        "reconst_thresh/3": "12Layer-VAE-MSE",
        "reconst_thresh/4": "ALIstyle-AE-BCE",
        "reconst_thresh/5": "ALIstyle-AE-MSE",
        "reconst_thresh/6": "ALIstyle-VAE-BCE",
        "reconst_thresh/7": "ALIstyle-VAE-MSE",
        "reconst_thresh/8": "DeepRes-AE-BCE",
        "reconst_thresh/9": "DeepRes-AE-MSE",
        "reconst_thresh/10": "ALIRes-AE-BCE",
        "reconst_thresh/11": "ALIRes-AE-MSE",
        "reconst_thresh/12": "ALIRes-VAE-BCE",
        "reconst_thresh/13": "ALIRes-VAE-MSE",
    }
    if type(d2_handles) == str:
        d2_handles = [d2_handles, ]
    use_case1_acc = []
    use_case1_auroc = []
    use_case1_auprc = []
    tpr = []

    method_handles = []
    d3_handles = []
    true_d2_handles = []
    for row in results:
        if not any([handle in row[2] for handle in d2_handles]):
            continue
        if row[0] in rec_error:
            token = rec_error[row[0]]
        else:
            token = row[0]
        if token not in method_handles:
            method_handles.append(token)
        ind_1 = method_handles.index(token)
        if row[2] not in true_d2_handles:
            true_d2_handles.append(row[2])
        ind_2 = true_d2_handles.index(row[2])
        if row[3] not in d3_handles:
            d3_handles.append(row[3])
        ind_3 = d3_handles.index(row[3])
        use_case1_acc.append((row[6], ind_1, ind_2, ind_3))
        use_case1_auroc.append((row[7], ind_1, ind_2, ind_3))
        use_case1_auprc.append((row[8], ind_1, ind_2, ind_3))
    uc1_acc = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))
    uc1_roc = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))
    uc1_prc = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))
    weights = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))

    for acc, roc, prc in zip(use_case1_acc, use_case1_auroc, use_case1_auprc):
        try:
            weights[prc[1], prc[2], prc[3]] += 1
            uc1_acc[acc[1], acc[2], acc[3]] = uc1_acc[acc[1], acc[2], acc[3]] * (weights[acc[1], acc[2], acc[3]] - 1) / weights[acc[1], acc[2], acc[3]]\
                                              + acc[0] /(weights[acc[1], acc[2], acc[3]])
            uc1_roc[roc[1], roc[2], roc[3]] = uc1_roc[roc[1], roc[2], roc[3]] * (weights[roc[1], roc[2], roc[3]] - 1) / weights[roc[1], roc[2], roc[3]]\
                                              + roc[0] /(weights[roc[1], roc[2], roc[3]])
            uc1_prc[prc[1], prc[2], prc[3]] = uc1_prc[prc[1], prc[2], prc[3]] * (weights[prc[1], prc[2], prc[3]] - 1) / weights[prc[1], prc[2], prc[3]]\
                                              + prc[0] /(weights[prc[1], prc[2], prc[3]])
        except:
            pass
    inds_0, inds_1, inds_2 = np.nonzero(np.array(weights == 0))

    for i in range(len(inds_0)):
        print(method_handles[inds_0[i]],", ", true_d2_handles[inds_1[i]], ",", d3_handles[inds_2[i]], ", has no entry")
    # method means
    n = weights[0].sum()
    uc1_accm = np.average(uc1_acc, axis=(1,2), weights=weights)
    uc1_rocm = np.average(uc1_roc, axis=(1,2), weights=weights)
    uc1_prcm = np.average(uc1_prc, axis=(1,2), weights=weights)

    uc1_accv = weighted_std(uc1_acc, weights, axis=(1,2))/np.sqrt(n)
    uc1_rocv = weighted_std(uc1_roc, weights, axis=(1,2))/np.sqrt(n)
    uc1_prcv = weighted_std(uc1_prc, weights, axis=(1,2))/np.sqrt(n)
    def proc_var(m, v):
        upper = []
        lower = []
        for n in range(m.shape[0]):
            lower.append(v[n])
            if m[n] + v[n] > 1.0:
                upper.append(1.0 - m[n])
            else:
                upper.append(v[n])
        return np.array([lower, upper])
    uc1_accv = proc_var(uc1_accm, uc1_accv)
    uc1_rocv = proc_var(uc1_rocm, uc1_rocv)
    uc1_prcv = proc_var(uc1_prcm, uc1_prcv)
    #uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=1)
    #uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=1)
    #uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=1)

    #accdelta = np.array((uc1_accq[1, :] - uc1_accm, uc1_accm - uc1_accq[0, :]))
    #rocdelta = np.array((uc1_rocq[1, :] - uc1_rocm, uc1_rocm - uc1_rocq[0, :]))
    #prcdelta = np.array((uc1_prcq[1, :] - uc1_prcm, uc1_prcm - uc1_prcq[0, :]))

    fig, ax = plt.subplots()
    ind = np.arange(len(uc1_accm))  # the x locations for the groups
    width = 0.25  # the width of the bars
    rects1 = ax.bar(ind - width * 0.75, uc1_accm, width, yerr=uc1_accv,
                    label='Accuracy')
    rects2 = ax.bar(ind, uc1_rocm, width, yerr=uc1_rocv,
                    label='AUROC')
    rects3 = ax.bar(ind + width * 0.75, uc1_prcm, width, yerr=uc1_prcv,
                    label='AUPRC')

    ax.set_xticks(ind)
    ax.set_xticklabels(method_handles, rotation=45, ha='right')
    title_str = title + ", D2="
    for handle in d2_handles:
        title_str += handle+', '
    title_str += "error bar shows standard error"
    ax.set_title(title_str)
    ax.legend()
    fig.set_size_inches(25, 9.5)
    # when saving, specify the DPI


    backend = matplotlib.get_backend()

    if backend == "QT" or backend == "Qt5Agg":
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    elif backend == "TkAgg":
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
    else:
        manager = plt.get_current_fig_manager()
        manager.frame.Maximize(True)
    plt.show()
    plt.savefig(filename, dpi=100)
    return

if __name__ == "__main__":
    dir_path = "workspace/experiments/chest_eval_results"
    res = []
    for root, dirs, files in os.walk(dir_path):
        print(files)
        res = files
    all_results = []
    for file in res:
        res_path = os.path.join(dir_path, file)
        results = torch.load(res_path)['results']
        all_results.extend(results)

    filenames = ["UC1_7seeds_se.png",
                 "UC2_7seeds_se.png",
                 "UC3_7seeds_se.png",
                 ]
    d2sets = [["CIFAR",
               "MURA",
               "Noise",
               'MNIST',
               'FashionMNIST',
               'NotMNIST',
               'CIFAR100',
               'CIFAR10',
               'STL10',
               'TinyImagenet',
               ],
              "PAD",
              "NIHCC",
              ]
    titles = ["Usecase 1", "Usecase 2", "Usecase 3"]
    for fn, d2, title in zip(filenames, d2sets, titles):
        make_plot(fn, d2, title, all_results)
