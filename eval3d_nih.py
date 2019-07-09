from __future__ import print_function

import os
from termcolor import colored
import torch

from utils.args import args
import global_vars as Global
from datasets.NIH_Chest import NIHChestBinaryTrainSplit, NIHChestBinaryValSplit, NIHChestBinaryTestSplit
from models.ALImodel import *

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

    test_acc, test_auroc = ODmethod.test_H(test_mixture)
    return test_acc, test_auroc


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
    for m, ds, dm, dt, mid, a1, a2, a3 in results['results']:
        if m == method and ds == d1 and dm == d2 and dt == d3:
            return True
    return False


RESULTS_VER = 0
if __name__ == '__main__':
    results_path = os.path.join(args.experiment_path, 'results.pth')
    results = init_and_load_results(results_path, args)
    methods = [
               'prob_threshold/0',  #  'prob_threshold/1',
               'score_svm/0',          #'score_svm/1',
               'openmax/0',            #'openmax/1',
               'binclass/0',           #'binclass/1',
               'odin/0',               #'odin/1',


        ]
    methods_64 = [
         'reconst_thresh/0',   #  'reconst_thresh/1',
         'ALI_reconst/0', 'ALI_reconst/1', #'ALI_reconst/0',
         'knn/1', 'knn/2', 'knn/4', 'knn/8',
          'vaeaeknn/1', 'mseaeknn/1',#'bceaeknn/1',
          'vaeaeknn/2', 'mseaeknn/2', #'bceaeknn/2',
          'vaeaeknn/4', 'mseaeknn/4', #'bceaeknn/4',
          'vaeaeknn/8', 'mseaeknn/8', #'bceaeknn/8',
    ]

    D1 = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, 'NIHCC'))
    D164 = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, 'NIHCC'), downsample=64)
    args.D1 = 'NIHCC'

    # Usecase 1 Evaluation
    D2 = Global.all_datasets['CIFAR10'](root_path=os.path.join(args.root_path, 'cifar10'))
    args.D2 = "CIFAR10"
    d3s = ['UniformNoise',
           'NormalNoise',
           'MNIST',
           'FashionMNIST',
           'NotMNIST',
           'CIFAR100',
           'STL10',
           'TinyImagenet',
           'MURA',
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
        mt = Global.get_method(method, args)
        if not all([has_done_before(method, 'NIHCC', 'CIFAR', d3) for d3 in d3s]):
            trainval_acc = train_subroutine(mt, D1, D2)
        for d3, D3 in zip(d3s,D3s):
            if not has_done_before(method, 'NIHCC', 'CIFAR', d3):
                test_acc, test_auroc = eval_subroutine(mt, D1, D3)
                results['results'].append((method, 'NIHCC', 'CIFAR', d3, mt.method_identifier(), trainval_acc, test_acc, test_auroc))
                torch.save(results, results_path)

    for method in methods_64:
        mt = Global.get_method(method, args)
        if not all([has_done_before(method, 'NIHCC', 'CIFAR', d3) for d3 in d3s]):
            trainval_acc = train_subroutine(mt, D164, D2)
        for d3, D3 in zip(d3s,D3s):
            if not has_done_before(method, 'NIHCC', 'CIFAR', d3):
                test_acc, test_auroc = eval_subroutine(mt, D164, D3)
                results['results'].append((method, 'NIHCC', 'CIFAR', d3, mt.method_identifier(), trainval_acc, test_acc, test_auroc))
                torch.save(results, results_path)
    # Usecase 3 Evaluation
    D2 = NIHChestBinaryValSplit(root_path=os.path.join(args.root_path, 'NIHCC'))
    D3 = NIHChestBinaryTestSplit(root_path=os.path.join(args.root_path, 'NIHCC'))

    args.D2 = 'NIHChest'
    for method in methods:
        mt = Global.get_method(method, args)
        if not has_done_before(method, 'NIHCC', 'NIHCC_val', 'NICC_test'):
            trainval_acc = train_subroutine(mt, D1, D2)
        test_acc, test_auroc = eval_subroutine(mt, D1, D3)
        results['results'].append((method, 'NIHCC', 'NIHCC_val', 'NICC_test', mt.method_identifier(), trainval_acc, test_acc, test_auroc))
        torch.save(results, results_path)

    for method in methods_64:
        mt = Global.get_method(method, args)
        if not has_done_before(method, 'NIHCC', 'NIHCC_val', 'NICC_test'):
            trainval_acc = train_subroutine(mt, D164, D2)
        test_acc, test_auroc = eval_subroutine(mt, D164, D3)
        results['results'].append((method, 'NIHCC', 'NIHCC_val', 'NICC_test', mt.method_identifier(), trainval_acc, test_acc, test_auroc))
        torch.save(results, results_path)

    for i, (m, ds, dm, dt, mi, a_train, a_test, auc_test) in enumerate(results['results']):
        print ('%d\t%s\t%15s\t%-15s\t%.2f%% / %.2f%% - %.2f%%'%(i, m, '%s-%s'%(ds, dm), dt, a_train*100, a_test*100, auc_test*100))
