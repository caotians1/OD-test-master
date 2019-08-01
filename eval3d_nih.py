from __future__ import print_function

import os
from termcolor import colored
import numpy as np
import torch

from utils.args import args
import global_vars as Global
from datasets.NIH_Chest import NIHChestBinaryTrainSplit, NIHChestBinaryValSplit, NIHChestBinaryTestSplit
from models.ALImodel import *
import matplotlib as mpl
mpl.rcParams['text.antialiased']=False
import matplotlib.pyplot as plt

def train_subroutine(ODmethod, D1, D2):
    d1_train = D1.get_D1_train()
    ODmethod.propose_H(d1_train)
    d1_valid = D1.get_D1_valid()
    d2_valid = D2.get_D2_valid(D1)
    d1_valid_len = len(d1_valid)
    d2_valid_len = len(d2_valid)
    final_len = min(d1_valid_len, d2_valid_len)
    print("Adjusting %s and %s to %s" % (colored('D1=%d' % d1_valid_len, 'red'),
                                         colored('D2=%d' % d2_valid_len, 'red'),
                                         colored('Min=%d' % final_len, 'green')))
    d1_valid.trim_dataset(final_len)
    d2_valid.trim_dataset(final_len)
    valid_mixture = d1_valid + d2_valid
    print("Final valid size: %d+%d=%d" % (len(d1_valid), len(d2_valid), len(valid_mixture)))
    train_acc = ODmethod.train_H(valid_mixture)
    return train_acc


def eval_subroutine(ODmethod, D1, D3):
    d1_test = D1.get_D1_test()
    d3_test = D3.get_D2_test(D1)
    # Adjust the sizes.
    d1_test_len = len(d1_test)
    d3_test_len = len(d3_test)
    final_len = min(d1_test_len, d3_test_len)
    print("Adjusting %s and %s to %s" % (colored('D1=%d' % d1_test_len, 'red'),
                                         colored('D2=%d' % d3_test_len, 'red'),
                                         colored('Min=%d' % final_len, 'green')))
    d1_test.trim_dataset(final_len)
    d3_test.trim_dataset(final_len)
    test_mixture = d1_test + d3_test
    print("Final test size: %d+%d=%d" % (len(d1_test), len(d3_test), len(test_mixture)))

    acc, auroc, auprc, fpr, tpr, precision, recall, TP, TN, FP, FN = ODmethod.test_H(test_mixture)
    '''fig = plt.figure()
    lw = 2
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, color='g',
             lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.axes.set_xlim([0.0, 1.0])
    ax.axes.set_ylim([0.0, 1.05])
    ax.axes.set_xlabel('False Positive Rate')
    ax.axes.set_ylabel('True Positive Rate')
    ax.axes.legend(loc="lower right")
    plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    ROC = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    ROC = ROC.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    fig = plt.figure()
    lw = 2
    ax = fig.add_subplot(111)
    ax.plot(recall, precision, color='b',
            lw=lw, label='P-R curve (AP = %0.2f)' % auprc)
    ax.axes.set_xlim([0.0, 1.0])
    ax.axes.set_ylim([0.0, 1.05])
    ax.axes.set_xlabel('Recall')
    ax.axes.set_ylabel('Precision')
    ax.axes.legend(loc="lower left")
    plt.setp([ax.get_xticklines() + ax.get_yticklines() + ax.get_xgridlines() + ax.get_ygridlines()], antialiased=False)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    PRC = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    PRC = PRC.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)'''
    return acc, auroc, auprc, None, None, fpr, tpr, precision, recall, TP, TN, FP, FN


def init_and_load_results(path, args):
    # If results exists already, just continue where left off.

    if os.path.exists(path) and not args.force_run:
        print("Loading previous checkpoint")
        results = torch.load(path)
        if type(results) is dict:
            if results['ver'] == RESULTS_VER:
                print("Loaded previous checkpoint")
                return results
    print("No compatible result found, initializing fresh results")
    return {'ver': RESULTS_VER, 'results':[]}


def has_done_before(method, d1, d2, d3):
    for res in results['results']:
        if res[0] == method and res[1] == d1 and res[2] == d2 and res[3] == d3:
            return True
    return False


RESULTS_VER = 4
if __name__ == '__main__':
    results_path = os.path.join(args.experiment_path, 'results.pth')
    results = init_and_load_results(results_path, args)
    methods = [
               'prob_threshold/0',  #'prob_threshold/1',
               'score_svm/0',          #'score_svm/1',
               'openmax/0',            #'openmax/1',
               'binclass/0',           #'binclass/1',
               'odin/0',              # 'odin/1',
                "Maha",

        ]
    methods_64 = [
          'reconst_thresh/0', 'reconst_thresh/1',   'reconst_thresh/2', 'reconst_thresh/3',
        'reconst_thresh/4', 'reconst_thresh/5', 'reconst_thresh/6', 'reconst_thresh/7',
        'reconst_thresh/8', 'reconst_thresh/9', 'reconst_thresh/10','reconst_thresh/11',
        'reconst_thresh/12', 'reconst_thresh/13',
         'ALI_reconst/0', #'ALI_reconst/1', #'ALI_reconst/0',
         'knn/1', 'knn/2', 'knn/4', 'knn/8',
          'vaemseaeknn/1','vaebceaeknn/1', 'mseaeknn/1', 'bceaeknn/1',
          'vaemseaeknn/2','vaebceaeknn/2', 'mseaeknn/2',  'bceaeknn/2',
          'vaemseaeknn/4','vaebceaeknn/4', 'mseaeknn/4',  'bceaeknn/4',
          'vaemseaeknn/8','vaebceaeknn/8', 'mseaeknn/8',  'bceaeknn/8',
    ]

    D1 = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, 'NIHCC'))
    D164 = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, 'NIHCC'), downsample=64)
    args.D1 = 'NIHCC'

    #Usecase 1 Evaluation
    d2s = [ 'CIFAR10', 'UniformNoise', 'MURAHAND', ]
    D2s = []
    for d2 in d2s:
        dataset = Global.all_datasets[d2]
        if 'dataset_path' in dataset.__dict__:
            print(os.path.join(args.root_path, dataset.dataset_path))
            D2s.append(dataset(root_path=os.path.join(args.root_path, dataset.dataset_path)))
        else:
            D2s.append(dataset())

    d3s = [
           'NormalNoise',
           'MNIST',
           'FashionMNIST',
           'NotMNIST',
           'CIFAR100',
           'STL10',
           'TinyImagenet',
           'MURAWRIST',
           'MURAELBOW',
           'MURAFINGER',
           'MURAFOREARM',
           'MURAHUMERUS',
           'MURASHOULDER',
           ]
    D3s=[]
    for d3 in d3s:
        dataset = Global.all_datasets[d3]
        if 'dataset_path' in dataset.__dict__:
            print(os.path.join(args.root_path, dataset.dataset_path))
            D3s.append(dataset(root_path=os.path.join(args.root_path, dataset.dataset_path)))
        else:
            D3s.append(dataset())

    for method in methods:
        print("current method", method)
        mt = Global.get_method(method, args)
        for d2, D2 in zip(d2s, D2s):
            if not all([has_done_before(method, 'NIHCC', d2, d3) for d3 in d3s]):
                args.D2 = d2
                trainval_acc = train_subroutine(mt, D1, D2)
                for d3, D3 in zip(d3s,D3s):
                    if (not has_done_before(method, 'NIHCC', d2, d3)) and (d2 != d3):
                        print("Evaluating: ", method, 'NIHCC', d2, d3)
                        test_results = eval_subroutine(mt, D1, D3)
                        results['results'].append([method, 'NIHCC', d2, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                        torch.save(results, results_path)

    for method in methods_64:
        print("current method", method)
        mt = Global.get_method(method, args)
        for d2, D2 in zip(d2s, D2s):
            if not all([has_done_before(method, 'NIHCC', d2, d3) for d3 in d3s]):
                args.D2 = d2
                trainval_acc = train_subroutine(mt, D164, D2)
                for d3, D3 in zip(d3s,D3s):
                    if (not has_done_before(method, 'NIHCC', d2, d3)) and (d2 != d3):
                        print("Evaluating: ", method, 'NIHCC', d2, d3)
                        test_results = eval_subroutine(mt, D164, D3)
                        results['results'].append([method, 'NIHCC', d2, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                        torch.save(results, results_path)

    # usecase 2

    d3s = ["PADChestAP",
            "PADChestL",
           "PADChestAPHorizontal",
           "PADChestPED"
           ]
    D3s = []
    for d3 in d3s:
        dataset = Global.all_datasets[d3]
        if 'dataset_path' in dataset.__dict__:
            print(os.path.join(args.root_path, dataset.dataset_path))
            D3s.append(dataset(root_path=os.path.join(args.root_path, dataset.dataset_path)))
        else:
            D3s.append(dataset())

    for method in methods:
        print("current method", method)
        for d2, D2 in zip(d3s, D3s):
            args.D2 = d2
            mt = Global.get_method(method, args)

            if not all([has_done_before(method, 'NIHCC', d2, d3) for d3 in d3s]):
                trainval_acc = train_subroutine(mt, D1, D2)
            for d3, D3 in zip(d3s,D3s):
                if d2 == d3:
                    continue
                if not has_done_before(method, 'NIHCC', d2, d3):
                    print("Evaluating: ", method, 'NIHCC', d2, d3)
                    test_results = eval_subroutine(mt, D1, D3)
                    results['results'].append([method, 'NIHCC', d2, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                    torch.save(results, results_path)

    for method in methods_64:
        print("current method", method)
        for d2, D2 in zip(d3s, D3s):
            args.D2 = d2
            mt = Global.get_method(method, args)
            if not all([has_done_before(method, 'NIHCC', d2, d3) for d3 in d3s]):
                trainval_acc = train_subroutine(mt, D164, D2)
            for d3, D3 in zip(d3s,D3s):
                if d2 == d3:
                    continue
                if not has_done_before(method, 'NIHCC', d2, d3):
                    print("Evaluating: ", method, 'NIHCC', d2, d3)
                    test_results = eval_subroutine(mt, D164, D3)
                    results['results'].append(
                        [method, 'NIHCC', d2, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                    torch.save(results, results_path)

    # Usecase 3 Evaluation
    D2 = NIHChestBinaryValSplit(root_path=os.path.join(args.root_path, 'NIHCC'))
    D3 = NIHChestBinaryTestSplit(root_path=os.path.join(args.root_path, 'NIHCC'))

    args.D2 = 'NIHChest'
    for method in methods:
        print("current method", method)
        mt = Global.get_method(method, args)
        if not has_done_before(method, 'NIHCC', 'NIHCC_val', 'NICC_test'):
            trainval_acc = train_subroutine(mt, D1, D2)
            print("Evaluating: ", method, 'NIHCC', 'NIHChest', 'NIHCC_test')
            test_results = eval_subroutine(mt, D1, D3)
            results['results'].append([method, 'NIHCC', 'NIHCC_val', 'NICC_test', mt.method_identifier(), trainval_acc]
                                        + list(test_results))

            torch.save(results, results_path)

    for method in methods_64:
        print("current method", method)
        mt = Global.get_method(method, args)
        if not has_done_before(method, 'NIHCC', 'NIHCC_val', 'NICC_test'):
            trainval_acc = train_subroutine(mt, D164, D2)
            print("Evaluating: ", method, 'NIHCC', 'NIHChest', 'NIHCC_test')
            test_results = eval_subroutine(mt, D164, D3)
            results['results'].append([method, 'NIHCC', 'NIHCC_val', 'NICC_test', mt.method_identifier(), trainval_acc]
                                      + list(test_results))
            torch.save(results, results_path)

    for i, (m, ds, dm, dt, mi, a_train, a_test, auc_test, AP_test, ROC, PRC, fpr, tpr, precision, recall, TP, TN, FP, FN) in enumerate(results['results']):
        print ('%d\t%s\t%15s\t%-15s\t%.2f%% / %.2f%% - %.2f%%'%(i, m, '%s-%s'%(ds, dm), dt, a_train*100, a_test*100, auc_test*100))
