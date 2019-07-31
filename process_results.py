import numpy as np
import csv
import torch
import matplotlib
import matplotlib.pyplot as plt

res_path = "workspace/experiments/eval3d_nih_nofig/results.pth"
results = torch.load(res_path)['results']

use_case1_acc = []
use_case1_auroc = []
use_case1_auprc = []
method_handles = []
d3_handles = []
for row in results:
    if not row[2] == 'CIFAR':
        continue
    if row[0] not in method_handles:
        method_handles.append(row[0])
    ind_1 = method_handles.index(row[0])
    if row[3] not in d3_handles:
        d3_handles.append(row[3])
    ind_2 = d3_handles.index(row[3])
    use_case1_acc.append((row[6], ind_1, ind_2))
    use_case1_auroc.append((row[7], ind_1, ind_2))
    use_case1_auprc.append((row[8], ind_1, ind_2))
uc1_acc = np.zeros((len(method_handles), len(d3_handles)))
uc1_roc = np.zeros((len(method_handles), len(d3_handles)))
uc1_prc = np.zeros((len(method_handles), len(d3_handles)))
for acc, roc, prc in zip(use_case1_acc, use_case1_auroc, use_case1_auprc):
    uc1_acc[acc[1],acc[2]] = acc[0]
    uc1_roc[roc[1], roc[2]] = roc[0]
    uc1_prc[prc[1], prc[2]] = prc[0]
#method means
uc1_accm = uc1_acc.mean(1)
uc1_rocm = uc1_roc.mean(1)
uc1_prcm = uc1_prc.mean(1)

uc1_accv = uc1_acc.std(1) / np.sqrt(len(uc1_acc))
uc1_rocv = uc1_roc.std(1) / np.sqrt(len(uc1_acc))
uc1_prcv  = uc1_prc.std(1) / np.sqrt(len(uc1_acc))

uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=1)
uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=1)
uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=1)

accdelta = np.array((uc1_accq[1,:] - uc1_accm, uc1_accm - uc1_accq[0,:]))
rocdelta = np.array((uc1_rocq[1,:] - uc1_rocm, uc1_rocm - uc1_rocq[0,:]))
prcdelta = np.array((uc1_prcq[1,:] - uc1_prcm, uc1_prcm - uc1_prcq[0,:]))

fig, ax = plt.subplots()
ind = np.arange(len(uc1_accm))  # the x locations for the groups
width = 0.25  # the width of the bars
rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=accdelta,
                label='Accuracy')
rects2 = ax.bar(ind , uc1_rocm, width, yerr=rocdelta,
                label='AUROC')
rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=prcdelta,
                label='AUROC')

ax.set_xticks(ind)
ax.set_xticklabels(method_handles,  rotation='vertical')
ax.set_title("Usecase 1, error bar shows 25-75 quantile")
ax.legend()

#dataset means
uc1_dataacc = uc1_acc.mean(0)
uc1_dataroc = uc1_roc.mean(0)
uc1_dataprc = uc1_prc.mean(0)



use_case2_acc = []
use_case2_auroc = []
use_case2_auprc = []
d2_handles = []
d3_handles = []
for row in results:
    if not 'PAD' in row[2]:
        continue
    ind_1 = method_handles.index(row[0])
    if row[2] not in d2_handles:
        d2_handles.append(row[2])
    ind_2 = d2_handles.index(row[2])
    if row[3] not in d3_handles:
        d3_handles.append(row[3])
    ind_3 = d3_handles.index(row[3])
    use_case2_acc.append((row[6], ind_1, ind_2, ind_3))
    use_case2_auroc.append((row[7], ind_1, ind_2, ind_3))
    use_case2_auprc.append((row[8], ind_1, ind_2, ind_3))
uc1_acc = np.zeros((len(method_handles), len(d2_handles), len(d3_handles)))
uc1_roc = np.zeros((len(method_handles), len(d2_handles), len(d3_handles)))
uc1_prc = np.zeros((len(method_handles), len(d2_handles), len(d3_handles)))
for acc, roc, prc in zip(use_case2_acc, use_case2_auroc, use_case2_auprc):
    uc1_acc[acc[1], acc[2], acc[3]] = acc[0]
    uc1_roc[roc[1], roc[2], roc[3]] = roc[0]
    uc1_prc[prc[1], prc[2], prc[3]] = prc[0]
#method means
uc1_accm = uc1_acc.mean(axis=(1,2))
uc1_rocm = uc1_roc.mean(axis=(1,2))
uc1_prcm = uc1_prc.mean(axis=(1,2))

uc1_accv = uc1_acc.std(axis=(1,2)) / np.sqrt(len(uc1_acc))
uc1_rocv = uc1_roc.std(axis=(1,2)) / np.sqrt(len(uc1_acc))
uc1_prcv  = uc1_prc.std(axis=(1,2)) / np.sqrt(len(uc1_acc))

uc1_accq = np.quantile(uc1_acc, [.25, .75], axis=(1,2))
uc1_rocq = np.quantile(uc1_roc, [.25, .75], axis=(1,2))
uc1_prcq = np.quantile(uc1_prc, [.25, .75], axis=(1,2))

accdelta = np.array((uc1_accq[1,:] - uc1_accm, uc1_accm - uc1_accq[0,:]))
rocdelta = np.array((uc1_rocq[1,:] - uc1_rocm, uc1_rocm - uc1_rocq[0,:]))
prcdelta = np.array((uc1_prcq[1,:] - uc1_prcm, uc1_prcm - uc1_prcq[0,:]))

fig, ax = plt.subplots()
ind = np.arange(len(uc1_accm))  # the x locations for the groups
width = 0.25  # the width of the bars
rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=accdelta,
                label='Accuracy')
rects2 = ax.bar(ind , uc1_rocm, width, yerr=rocdelta,
                label='AUROC')
rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=prcdelta,
                label='AUROC')

ax.set_xticks(ind)
ax.set_xticklabels(method_handles,  rotation='vertical')
ax.set_title("Usecase 2, error bar shows 25-75 quantile")
ax.legend()


fig, ax = plt.subplots()
ind = np.arange(len(uc1_accm))  # the x locations for the groups
width = 0.25  # the width of the bars
rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=uc1_accv,
                label='Accuracy')
rects2 = ax.bar(ind , uc1_rocm, width, yerr=uc1_rocv,
                label='AUROC')
rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=uc1_prcv,
                label='AUROC')

ax.set_xticks(ind)
ax.set_xticklabels(method_handles,  rotation='vertical')
ax.set_title("Usecase 2, error bar shows stderr")
ax.legend()



use_case2_acc = []
use_case2_auroc = []
use_case2_auprc = []

for row in results:
    if not 'NIHCC' in row[2]:
        continue
    ind_1 = method_handles.index(row[0])

    use_case2_acc.append((row[6], ind_1,))
    use_case2_auroc.append((row[7], ind_1))
    use_case2_auprc.append((row[8], ind_1))
uc1_acc = np.zeros((len(method_handles)))
uc1_roc = np.zeros((len(method_handles)))
uc1_prc = np.zeros((len(method_handles)))
for acc, roc, prc in zip(use_case2_acc, use_case2_auroc, use_case2_auprc):
    uc1_acc[acc[1]] = acc[0]
    uc1_roc[roc[1]] = roc[0]
    uc1_prc[prc[1]] = prc[0]
#method means
uc1_accm = uc1_acc
uc1_rocm = uc1_roc
uc1_prcm = uc1_prc

fig, ax = plt.subplots()
ind = np.arange(len(uc1_accm))  # the x locations for the groups
width = 0.25  # the width of the bars
rects1 = ax.bar(ind - width*0.75, uc1_accm, width,
                label='Accuracy')
rects2 = ax.bar(ind , uc1_rocm, width,
                label='AUROC')
rects3 = ax.bar(ind + width*0.75, uc1_rocm, width,
                label='AUROC')

ax.set_xticks(ind)
ax.set_xticklabels(method_handles,  rotation='vertical')
ax.set_title("Usecase 3, (i didn't record performance per outlier class...)")
ax.legend()


fig, ax = plt.subplots()
ind = np.arange(len(uc1_accm))  # the x locations for the groups
width = 0.25  # the width of the bars
rects1 = ax.bar(ind - width*0.75, uc1_accm, width, yerr=uc1_accv,
                label='Accuracy')
rects2 = ax.bar(ind , uc1_rocm, width, yerr=uc1_rocv,
                label='AUROC')
rects3 = ax.bar(ind + width*0.75, uc1_rocm, width, yerr=uc1_prcv,
                label='AUROC')

ax.set_xticks(ind)
ax.set_xticklabels(method_handles,  rotation='vertical')
ax.set_title("Usecase 2, error bar shows stderr")
ax.legend()