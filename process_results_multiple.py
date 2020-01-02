import numpy as np
import csv, os
import torch
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import _pickle

def weighted_std(values, weights, axis=None):
    if axis is None:
        axis = np.arange(len(values.shape))

    average = np.average(values, weights=weights, axis=axis)
    for axi in axis:
        average = np.expand_dims(average, axi)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return np.sqrt(variance)

def process_results(d2_handles, results):
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
    __acc = []
    __auroc = []
    __auprc = []

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
        __acc.append((row[6], ind_1, ind_2, ind_3))
        __auroc.append((row[7], ind_1, ind_2, ind_3))
        __auprc.append((row[8], ind_1, ind_2, ind_3))
    uc_acc = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))
    uc_roc = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))
    uc_prc = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))
    weights = np.zeros((len(method_handles), len(true_d2_handles), len(d3_handles)))

    for acc, roc, prc in zip(__acc, __auroc, __auprc):
        try:
            weights[prc[1], prc[2], prc[3]] += 1
            uc_acc[acc[1], acc[2], acc[3]] = uc_acc[acc[1], acc[2], acc[3]] * (weights[acc[1], acc[2], acc[3]] - 1) / \
                                             weights[acc[1], acc[2], acc[3]] \
                                             + acc[0] / (weights[acc[1], acc[2], acc[3]])
            uc_roc[roc[1], roc[2], roc[3]] = uc_roc[roc[1], roc[2], roc[3]] * (weights[roc[1], roc[2], roc[3]] - 1) / \
                                             weights[roc[1], roc[2], roc[3]] \
                                             + roc[0] / (weights[roc[1], roc[2], roc[3]])
            uc_prc[prc[1], prc[2], prc[3]] = uc_prc[prc[1], prc[2], prc[3]] * (weights[prc[1], prc[2], prc[3]] - 1) / \
                                             weights[prc[1], prc[2], prc[3]] \
                                             + prc[0] / (weights[prc[1], prc[2], prc[3]])
        except:
            pass
    inds_0, inds_1, inds_2 = np.nonzero(np.array(weights == 0))

    for i in range(len(inds_0)):
        print(method_handles[inds_0[i]], ", ", true_d2_handles[inds_1[i]], ",", d3_handles[inds_2[i]], ", has no entry")

    return np.stack([weights, uc_acc, uc_roc, uc_prc]), [method_handles, true_d2_handles, d3_handles]


def make_plot(filename, title, true_d2_handles, method_handles, csv_data):
    weights = csv_data[0]
    uc_acc = csv_data[1]
    uc_roc = csv_data[2]
    uc_prc = csv_data[3]
    uc1_accm = np.average(uc_acc, axis=(1,2), weights=weights)
    uc1_rocm = np.average(uc_roc, axis=(1,2), weights=weights)
    uc1_prcm = np.average(uc_prc, axis=(1,2), weights=weights)

    uc1_accv = weighted_std(uc_acc, weights, axis=(1,2))
    uc1_rocv = weighted_std(uc_roc, weights, axis=(1,2))
    uc1_prcv = weighted_std(uc_prc, weights, axis=(1,2))
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
    for handle in true_d2_handles:
        title_str += handle+', '
    title_str += "error bar shows std"
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
    #else:
    #    manager = plt.get_current_fig_manager()
    #    manager.frame.Maximize(True)
    plt.show()
    plt.savefig(filename, dpi=300)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, help="save images and csvs here")
    parser.add_argument("--result_dir", type=str, default="workspace/experiments/padchest_res_raw", help="path to folder of result_n.pth files")
    parser.add_argument("--cache_dir", type=str, help="load cached results from here (precedes results path when specified)")
    parser.add_argument("--dataset", default="PAD", help="PAD or NIHCC")
    args = parser.parse_args()
    if args.dataset == "PAD":
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
                  ['PADChestAPHorizontal',
                   'PADChestPED',
                   'PADChestAP',
                   'PADChestPA',
                   ],
                  ['PADChestL_cardiomegaly',
                   'PADChestL_pneumothorax',
                   'PADChestL_nodule',
                   'PADChestL_mass',
                   ]
                  ]
    elif args.dataset == "NIHCC":
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
                  ["PAD",],
                  ["NIHCC",],
                  ]
    if args.cache_dir is None:
        dir_path = args.result_dir
        res = []
        for root, dirs, files in os.walk(dir_path):
            print(files)
            res = files
        all_results = []
        for file in res:
            res_path = os.path.join(dir_path, file)
            results = torch.load(res_path)['results']
            all_results.extend(results)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for i, d2tags in zip(range(1,4,1), d2sets):
        if args.cache_dir is None:
            csv_data, csv_headers = process_results(d2tags, all_results)
            np.save(os.path.join(args.output_dir, "data_UC%d_%s.npy" % (i, args.dataset)), csv_data)
            with open(os.path.join(args.output_dir, "headers_UC%d_%s.pkl" % (i, args.dataset)), "wb") as fp:
                _pickle.dump(csv_headers, fp)
            csv_list = [["D1", "Method", "D2", "D3", "multiplicity", "Accuracy (%)", "AUROC (%)", "AUPRC (%)"],]
            for m in range(csv_data.shape[1]):
                for d2 in range(csv_data.shape[2]):
                    for d3 in range(csv_data.shape[3]):
                        if int(csv_data[0, m, d2, d3]) == 0:
                            continue
                        csv_list.append([args.dataset, csv_headers[0][m], csv_headers[1][d2], csv_headers[2][d3],
                                         int(csv_data[0, m, d2, d3]),   #weight
                                         csv_data[1, m, d2, d3]*100.,   #acc
                                         csv_data[2, m, d2, d3]*100.,   #auc
                                         csv_data[3, m, d2, d3]*100.,    #prc
                                         ])
            with open(os.path.join(args.output_dir, "data_UC%d_%s.csv" % (i, args.dataset)), "w+") as fp:
                writer = csv.writer(fp)
                for row in csv_list:
                    writer.writerow(row)

        else:
            assert os.path.isfile(os.path.join(args.cache_dir, "data_UC%d_%s.npy" % (i, args.dataset)))
            csv_data = np.load(os.path.join(args.cache_dir, "data_UC%d_%s.npy" % (i, args.dataset)))
            assert os.path.isfile(os.path.join(args.cache_dir, "headers_UC%d_%s.pkl" % (i, args.dataset)))
            with open(os.path.join(args.cache_dir, "headers_UC%d_%s.pkl" % (i, args.dataset)), "rb") as fp:
                csv_headers = _pickle.load(fp)

        fn = os.path.join(args.output_dir, "UC%d_%s.png" % (i, args.dataset))

        make_plot(fn, "Usecase %d" % i, d2tags, csv_headers[0], csv_data)





