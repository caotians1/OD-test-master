import numpy as np
import csv
import torch
import matplotlib
import matplotlib.pyplot as plt

res_path = "workspace/experiments/eval3d_nih_nofig_no3264_seed10/results.pth"
results = torch.load(res_path)['results']
rec_error = {
            "reconst_thresh/0":"12Layer-AE-BCE",
            "reconst_thresh/1":"12Layer-AE-MSE",
            "reconst_thresh/2":"12Layer-VAE-BCE",
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
def weighted_std(values, weights, axis=None):
    if axis is None:
        axis = np.arange(len(values.shape))

    average = np.average(values, weights=weights, axis=axis)
    for axi in axis:
        average = np.expand_dims(average, axi)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)

def make_plot(filename, d2_handles, title):
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
            uc1_acc[acc[1], acc[2], acc[3]] = acc[0]
            uc1_roc[roc[1], roc[2], roc[3]] = roc[0]
            uc1_prc[prc[1], prc[2], prc[3]] = prc[0]
            weights[prc[1], prc[2], prc[3]] = 1
        except:
            pass
    inds_0, inds_1, inds_2 = np.nonzero(np.array(weights == 0))

    for i in range(len(inds_0)):
        print(method_handles[inds_0[i]],", ", true_d2_handles[inds_1[i]], ",", d3_handles[inds_2[i]], ", has no entry")
    # method means
    uc1_accm = np.average(uc1_acc, axis=(1,2), weights=weights)
    uc1_rocm = np.average(uc1_roc, axis=(1,2), weights=weights)
    uc1_prcm = np.average(uc1_prc, axis=(1,2), weights=weights)

    uc1_accv = weighted_std(uc1_acc, weights, axis=(1,2))
    uc1_rocv = weighted_std(uc1_roc, weights, axis=(1,2))
    uc1_prcv = weighted_std(uc1_prc, weights, axis=(1,2))

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
    rects3 = ax.bar(ind + width * 0.75, uc1_rocm, width, yerr=uc1_prcv,
                    label='AUPRC')

    ax.set_xticks(ind)
    ax.set_xticklabels(method_handles, rotation=45, ha='right')
    title_str = title + ", D2="
    for handle in d2_handles:
        title_str += handle+', '
    title_str += "error bar shows std"
    ax.set_title(title_str)
    ax.legend()
    fig.set_size_inches(25, 9.5)
    # when saving, specify the DPI


    backend = matplotlib.get_backend()

    if backend == "QT":
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
    elif backend == "TkAgg":
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
    else:
        manager = plt.get_current_fig_manager()
        manager.frame.Maximize(True)
    #plt.show()
    plt.savefig(filename, dpi=100)
    return

filenames = ["UC1_224-64_2.png",
             "UC2_224-64_2.png",
             "UC3_224-64_2.png",
             ]
d2sets = [["CIFAR10", "MURAHAND", "UniformNoise",],
          "PAD", "NIHCC"]
titles = ["Usecase 1", "Usecase 2", "Usecase 3"]
for fn, d2, title in zip(filenames, d2sets, titles):
    make_plot(fn, d2, title)
#
# use_case1_acc = []
# use_case1_auroc = []
# use_case1_auprc = []
# method_handles = []
# d3_handles = []
# for row in results:
#     if not row[2] == 'CIFAR10':# or row[2] == 'MURAHAND' or row[2] == "UniformNoise":
#         continue
#     if row[0] not in method_handles:
#         method_handles.append(row[0])
#     ind_1 = method_handles.index(row[0])
#     if row[3] not in d3_handles:
#         d3_handles.append(row[3])
#     ind_2 = d3_handles.index(row[3])
#     use_case1_acc.append((row[6], ind_1, ind_2))
#     use_case1_auroc.append((row[7], ind_1, ind_2))
#     use_case1_auprc.append((row[8], ind_1, ind_2))
# uc1_acc = np.zeros((len(method_handles), len(d3_handles)))
# uc1_roc = np.zeros((len(method_handles), len(d3_handles)))
# uc1_prc = np.zeros((len(method_handles), len(d3_handles)))
# for acc, roc, prc in zip(use_case1_acc, use_case1_auroc, use_case1_auprc):
#     uc1_acc[acc[1],acc[2]] = acc[0]
#     uc1_roc[roc[1], roc[2]] = roc[0]
#     uc1_prc[prc[1], prc[2]] = prc[0]
# #method means
# uc1_accm = uc1_acc.mean(1)
# uc1_rocm = uc1_roc.mean(1)
# uc1_prcm = uc1_prc.mean(1)
#
# uc1_accv = uc1_acc.std(1) / np.sqrt(len(uc1_acc))
# uc1_rocv = uc1_roc.std(1) / np.sqrt(len(uc1_acc))
# uc1_prcv  = uc1_prc.std(1) / np.sqrt(len(uc1_acc))
#
# uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=1)
# uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=1)
# uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=1)
#
# accdelta = np.array((uc1_accq[1,:] - uc1_accm, uc1_accm - uc1_accq[0,:]))
# rocdelta = np.array((uc1_rocq[1,:] - uc1_rocm, uc1_rocm - uc1_rocq[0,:]))
# prcdelta = np.array((uc1_prcq[1,:] - uc1_prcm, uc1_prcm - uc1_prcq[0,:]))
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=accdelta,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width, yerr=rocdelta,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=prcdelta,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 1, D1=CIFAR, error bar shows 25-75 quantile")
# ax.legend()
#
# use_case1_acc = []
# use_case1_auroc = []
# use_case1_auprc = []
# method_handles = []
# d3_handles = []
# for row in results:
#     if not row[2] == 'MURAHAND':# or row[2] == 'MURAHAND' or row[2] == "UniformNoise":
#         continue
#     if row[0] not in method_handles:
#         method_handles.append(row[0])
#     ind_1 = method_handles.index(row[0])
#     if row[3] not in d3_handles:
#         d3_handles.append(row[3])
#     ind_2 = d3_handles.index(row[3])
#     use_case1_acc.append((row[6], ind_1, ind_2))
#     use_case1_auroc.append((row[7], ind_1, ind_2))
#     use_case1_auprc.append((row[8], ind_1, ind_2))
# uc1_acc = np.zeros((len(method_handles), len(d3_handles)))
# uc1_roc = np.zeros((len(method_handles), len(d3_handles)))
# uc1_prc = np.zeros((len(method_handles), len(d3_handles)))
# for acc, roc, prc in zip(use_case1_acc, use_case1_auroc, use_case1_auprc):
#     uc1_acc[acc[1],acc[2]] = acc[0]
#     uc1_roc[roc[1], roc[2]] = roc[0]
#     uc1_prc[prc[1], prc[2]] = prc[0]
# #method means
# uc1_accm = uc1_acc.mean(1)
# uc1_rocm = uc1_roc.mean(1)
# uc1_prcm = uc1_prc.mean(1)
#
# uc1_accv = uc1_acc.std(1) / np.sqrt(len(uc1_acc))
# uc1_rocv = uc1_roc.std(1) / np.sqrt(len(uc1_acc))
# uc1_prcv  = uc1_prc.std(1) / np.sqrt(len(uc1_acc))
#
# uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=1)
# uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=1)
# uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=1)
#
# accdelta = np.array((uc1_accq[1,:] - uc1_accm, uc1_accm - uc1_accq[0,:]))
# rocdelta = np.array((uc1_rocq[1,:] - uc1_rocm, uc1_rocm - uc1_rocq[0,:]))
# prcdelta = np.array((uc1_prcq[1,:] - uc1_prcm, uc1_prcm - uc1_prcq[0,:]))
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=accdelta,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width, yerr=rocdelta,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=prcdelta,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 1, D2=MURA, error bar shows 25-75 quantile")
# ax.legend()
#
# use_case1_acc = []
# use_case1_auroc = []
# use_case1_auprc = []
# method_handles = []
# d3_handles = []
# for row in results:
#     if not row[2] == 'UniformNoise':# or row[2] == 'MURAHAND' or row[2] == "UniformNoise":
#         continue
#     if row[0] not in method_handles:
#         method_handles.append(row[0])
#     ind_1 = method_handles.index(row[0])
#     if row[3] not in d3_handles:
#         d3_handles.append(row[3])
#     ind_2 = d3_handles.index(row[3])
#     use_case1_acc.append((row[6], ind_1, ind_2))
#     use_case1_auroc.append((row[7], ind_1, ind_2))
#     use_case1_auprc.append((row[8], ind_1, ind_2))
# uc1_acc = np.zeros((len(method_handles), len(d3_handles)))
# uc1_roc = np.zeros((len(method_handles), len(d3_handles)))
# uc1_prc = np.zeros((len(method_handles), len(d3_handles)))
# for acc, roc, prc in zip(use_case1_acc, use_case1_auroc, use_case1_auprc):
#     uc1_acc[acc[1],acc[2]] = acc[0]
#     uc1_roc[roc[1], roc[2]] = roc[0]
#     uc1_prc[prc[1], prc[2]] = prc[0]
# #method means
# uc1_accm = uc1_acc.mean(1)
# uc1_rocm = uc1_roc.mean(1)
# uc1_prcm = uc1_prc.mean(1)
#
# uc1_accv = uc1_acc.std(1) / np.sqrt(len(uc1_acc))
# uc1_rocv = uc1_roc.std(1) / np.sqrt(len(uc1_acc))
# uc1_prcv  = uc1_prc.std(1) / np.sqrt(len(uc1_acc))
#
# uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=1)
# uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=1)
# uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=1)
#
# accdelta = np.array((uc1_accq[1,:] - uc1_accm, uc1_accm - uc1_accq[0,:]))
# rocdelta = np.array((uc1_rocq[1,:] - uc1_rocm, uc1_rocm - uc1_rocq[0,:]))
# prcdelta = np.array((uc1_prcq[1,:] - uc1_prcm, uc1_prcm - uc1_prcq[0,:]))
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=accdelta,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width, yerr=rocdelta,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=prcdelta,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 1, D2=UniformNoise, error bar shows 25-75 quantile")
# ax.legend()
#
# #dataset means
# uc1_dataacc = uc1_acc.mean(0)
# uc1_dataroc = uc1_roc.mean(0)
# uc1_dataprc = uc1_prc.mean(0)
#
#
#
# use_case2_acc = []
# use_case2_auroc = []
# use_case2_auprc = []
# d2_handles = []
# d3_handles = []
# for row in results:
#     if not 'PAD' in row[2]:
#         continue
#     ind_1 = method_handles.index(row[0])
#     if row[2] not in d2_handles:
#         d2_handles.append(row[2])
#     ind_2 = d2_handles.index(row[2])
#     if row[3] not in d3_handles:
#         d3_handles.append(row[3])
#     ind_3 = d3_handles.index(row[3])
#     use_case2_acc.append((row[6], ind_1, ind_2, ind_3))
#     use_case2_auroc.append((row[7], ind_1, ind_2, ind_3))
#     use_case2_auprc.append((row[8], ind_1, ind_2, ind_3))
# uc1_acc = np.zeros((len(method_handles), len(d2_handles), len(d3_handles)))
# uc1_roc = np.zeros((len(method_handles), len(d2_handles), len(d3_handles)))
# uc1_prc = np.zeros((len(method_handles), len(d2_handles), len(d3_handles)))
# for acc, roc, prc in zip(use_case2_acc, use_case2_auroc, use_case2_auprc):
#     uc1_acc[acc[1], acc[2], acc[3]] = acc[0]
#     uc1_roc[roc[1], roc[2], roc[3]] = roc[0]
#     uc1_prc[prc[1], prc[2], prc[3]] = prc[0]
# #method means
# uc1_accm = uc1_acc.mean(axis=(1,2))
# uc1_rocm = uc1_roc.mean(axis=(1,2))
# uc1_prcm = uc1_prc.mean(axis=(1,2))
#
# uc1_accv = uc1_acc.std(axis=(1,2)) / np.sqrt(len(uc1_acc))
# uc1_rocv = uc1_roc.std(axis=(1,2)) / np.sqrt(len(uc1_acc))
# uc1_prcv  = uc1_prc.std(axis=(1,2)) / np.sqrt(len(uc1_acc))
#
# uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=(1,2))
# uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=(1,2))
# uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=(1,2))
#
# accdelta = np.array((uc1_accq[1,:] - uc1_accm, uc1_accm - uc1_accq[0,:]))
# rocdelta = np.array((uc1_rocq[1,:] - uc1_rocm, uc1_rocm - uc1_rocq[0,:]))
# prcdelta = np.array((uc1_prcq[1,:] - uc1_prcm, uc1_prcm - uc1_prcq[0,:]))
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=accdelta,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width, yerr=rocdelta,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=prcdelta,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 2, error bar shows 25-75 quantile")
# ax.legend()
#
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=uc1_accv,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width, yerr=uc1_rocv,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=uc1_prcv,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 2, error bar shows stderr")
# ax.legend()
#
#
#
# use_case2_acc = []
# use_case2_auroc = []
# use_case2_auprc = []
#
# for row in results:
#     if not 'NIHCC' in row[2]:
#         continue
#     ind_1 = method_handles.index(row[0])
#
#     use_case2_acc.append((row[6], ind_1,))
#     use_case2_auroc.append((row[7], ind_1))
#     use_case2_auprc.append((row[8], ind_1))
# uc1_acc = np.zeros((len(method_handles)))
# uc1_roc = np.zeros((len(method_handles)))
# uc1_prc = np.zeros((len(method_handles)))
# for acc, roc, prc in zip(use_case2_acc, use_case2_auroc, use_case2_auprc):
#     uc1_acc[acc[1]] = acc[0]
#     uc1_roc[roc[1]] = roc[0]
#     uc1_prc[prc[1]] = prc[0]
# #method means
# uc1_accm = uc1_acc
# uc1_rocm = uc1_roc
# uc1_prcm = uc1_prc
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 3, (i didn't record performance per outlier class...)")
# ax.legend()
#
#
# fig, ax = plt.subplots()
# ind = np.arange(len(uc1_accm))  # the x locations for the groups
# width = 0.25  # the width of the bars
# rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=uc1_accv,
#                 label='Accuracy')
# rects2 = ax.bar(ind , uc1_rocm, width, yerr=uc1_rocv,
#                 label='AUROC')
# rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=uc1_prcv,
#                 label='AUROC')
#
# ax.set_xticks(ind)
# ax.set_xticklabels(method_handles,  rotation='vertical')
# ax.set_title("Usecase 2, error bar shows stderr")
# ax.legend()