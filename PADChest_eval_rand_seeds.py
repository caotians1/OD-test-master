from __future__ import print_function

import os
from termcolor import colored
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import torch

from utils.args import args
import global_vars as Global
from datasets.NIH_Chest import NIHChestBinaryTrainSplit, NIHChestBinaryValSplit, NIHChestBinaryTestSplit
from datasets.PADChest import PADChestBinaryTrainSplit, PADChestBinaryValSplit, PADChestBinaryTestSplit, PADChestSV
from models.ALImodel import *
import matplotlib as mpl
mpl.rcParams['text.antialiased']=False
import matplotlib.pyplot as plt
import random
from datasets.NIH_Chest import NIHChest

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
               "Maha1layer",
               "svknn",

        ]
    methods_64 = [
          'reconst_thresh/0', 'reconst_thresh/1',   'reconst_thresh/2', 'reconst_thresh/3',
         'reconst_thresh/4', 'reconst_thresh/5', 'reconst_thresh/6', 'reconst_thresh/7',
        #'reconst_thresh/8', 'reconst_thresh/9', 'reconst_thresh/10','reconst_thresh/11',
        #'reconst_thresh/12', 'reconst_thresh/13',
         #'ALI_reconst/0', #'ALI_reconst/1', #'ALI_reconst/0',
        #'aliknnsvm/1','aliknnsvm/8',
         'knn/1', 'knn/2', 'knn/4', 'knn/8',
          'vaemseaeknn/1','vaebceaeknn/1', 'mseaeknn/1', 'bceaeknn/1',
          'vaemseaeknn/8','vaebceaeknn/8', 'mseaeknn/8',  'bceaeknn/8',
        'alivaemseaeknn/1', 'alivaebceaeknn/1', 'alimseaeknn/1', 'alibceaeknn/1',
        'alivaemseaeknn/8', 'alivaebceaeknn/8', 'alimseaeknn/8', 'alibceaeknn/8',
    ]

    D1 = PADChestBinaryTrainSplit(root_path=os.path.join(args.root_path, 'PADChest'), binary=True)
    D164 = PADChestBinaryTrainSplit(root_path=os.path.join(args.root_path, "PADChest"), binary=True, downsample=64)
    args.D1 = 'PADChest'

    All_ODs = [
        'UniformNoise',
        'NormalNoise',
        'MNIST',
        'FashionMNIST',
        'NotMNIST',
        'CIFAR100',
        'CIFAR10',
        'STL10',
        'TinyImagenet',
        #'MURAHAND',
        #'MURAWRIST',
        #'MURAELBOW',
        #'MURAFINGER',
        #'MURAFOREARM',
        #'MURAHUMERUS',
        #'MURASHOULDER',
    ]


    random.shuffle(All_ODs)

    d2s = All_ODs[:3]
    d3s = All_ODs[3:]
    print("Chosen D2s: ", d2s)

    D2s = []
    for d2 in d2s:
        dataset = Global.all_datasets[d2]
        if 'dataset_path' in dataset.__dict__:
            print(os.path.join(args.root_path, dataset.dataset_path))
            D2s.append(dataset(root_path=os.path.join(args.root_path, dataset.dataset_path)))
        else:
            D2s.append(dataset())
    composite_D2 = D2s[0]
    for D2 in D2s[1:]:
        composite_D2 += D2

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
        if not all([has_done_before(method, 'PADChestL', composite_D2.name, d3) for d3 in d3s]):
            args.D2 = composite_D2.name
            trainval_acc = train_subroutine(mt, D1, composite_D2)
            for d3, D3 in zip(d3s,D3s):
                if not has_done_before(method, 'PADChestL', composite_D2.name, d3):
                    print("Evaluating: ", method, 'PADChestL', composite_D2.name, d3)
                    test_results = eval_subroutine(mt, D1, D3)
                    results['results'].append([method, 'PADChestL', composite_D2.name, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                    torch.save(results, results_path)

    for method in methods_64:
        print("current method", method)
        mt = Global.get_method(method, args)

        if not all([has_done_before(method, 'PADChestL', composite_D2.name, d3) for d3 in d3s]):
            args.D2 = composite_D2.name
            trainval_acc = train_subroutine(mt, D164, composite_D2)
            for d3, D3 in zip(d3s, D3s):
                if not has_done_before(method, 'PADChestL', composite_D2.name, d3):
                    print("Evaluating: ", method, 'PADChestL', composite_D2.name, d3)
                    test_results = eval_subroutine(mt, D164, D3)
                    results['results'].append(
                        [method, 'PADChestL', composite_D2.name, d3, mt.method_identifier(), trainval_acc] + list(
                            test_results))
                    torch.save(results, results_path)

    # usecase 2

    d3s = ["PADChestAP",
            "PADChestPA",
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

            if not all([has_done_before(method, 'PADChestL', d2, d3) for d3 in d3s]):
                trainval_acc = train_subroutine(mt, D1, D2)
            for d3, D3 in zip(d3s,D3s):
                if d2 == d3:
                    continue
                if not has_done_before(method, 'PADChestL', d2, d3):
                    print("Evaluating: ", method, 'PADChestL', d2, d3)
                    test_results = eval_subroutine(mt, D1, D3)
                    results['results'].append([method, 'PADChestL', d2, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                    torch.save(results, results_path)

    for method in methods_64:
        print("current method", method)
        for d2, D2 in zip(d3s, D3s):
            args.D2 = d2
            mt = Global.get_method(method, args)
            if not all([has_done_before(method, 'PADChestL', d2, d3) for d3 in d3s]):
                trainval_acc = train_subroutine(mt, D164, D2)
            for d3, D3 in zip(d3s,D3s):
                if d2 == d3:
                    continue
                if not has_done_before(method, 'PADChestL', d2, d3):
                    print("Evaluating: ", method, 'PADChestL', d2, d3)
                    test_results = eval_subroutine(mt, D164, D3)
                    results['results'].append(
                        [method, 'PADChestL', d2, d3, mt.method_identifier(), trainval_acc] + list(test_results))
                    torch.save(results, results_path)

    # Usecase 3 Evaluation

    d3_tags = ['cardiomegaly', 'pneumothorax', 'nodule', 'mass']

    args.D2 = 'PADChest'
    for method in methods:
        print("current method", method)
        for d2 in d3_tags:
            if not has_done_before(method, 'PADChestL', 'PADChestL_' + d2, 'PADChestL_test'):
                mt = Global.get_method(method, args)

                d3s = d3_tags.copy().remove(d2)

                D2 = PADChestSV(root_path=os.path.join(args.root_path, 'PADChest'), binary=True, test_length=5000,
                                keep_in_classes=[d2,])
                D3 = PADChestSV(root_path=os.path.join(args.root_path, 'PADChest'), binary=True, test_length=5000,
                                keep_in_classes=d3s)

                trainval_acc = train_subroutine(mt, D1, D2)
                print("Evaluating: ", method, 'PADChestL', 'PADChestL_' + d2, 'PADChestL_test')
                test_results = eval_subroutine(mt, D1, D3)
                results['results'].append([method, 'PADChestL', 'PADChestL_' + d2, 'PADChestL_test', mt.method_identifier(), trainval_acc]
                                            + list(test_results))

                torch.save(results, results_path)

    for method in methods_64:
        print("current method", method)
        for d2 in d3_tags:
            if not has_done_before(method, 'PADChestL', 'PADChestL_' + d2, 'PADChestL_test'):
                mt = Global.get_method(method, args)

                d3s = d3_tags.copy().remove(d2)

                D2 = PADChestSV(root_path=os.path.join(args.root_path, 'PADChest'), binary=True, test_length=5000,
                                keep_in_classes=[d2, ], downsample=64)
                D3 = PADChestSV(root_path=os.path.join(args.root_path, 'PADChest'), binary=True, test_length=5000,
                                keep_in_classes=d3s, downsample=64)

                trainval_acc = train_subroutine(mt, D164, D2)
                print("Evaluating: ", method, 'PADChestL', 'PADChestL_' + d2, 'PADChestL_test')
                test_results = eval_subroutine(mt, D164, D3)
                results['results'].append(
                    [method, 'PADChestL', 'PADChestL_' + d2, 'PADChestL_test', mt.method_identifier(), trainval_acc]
                    + list(test_results))

                torch.save(results, results_path)

    for i, (m, ds, dm, dt, mi, a_train, a_test, auc_test, AP_test, ROC, PRC, fpr, tpr, precision, recall, TP, TN, FP, FN) in enumerate(results['results']):
        print ('%d\t%s\t%15s\t%-15s\t%.2f%% / %.2f%% - %.2f%%'%(i, m, '%s-%s'%(ds, dm), dt, a_train*100, a_test*100, auc_test*100))
