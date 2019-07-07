from __future__ import print_function
import os,sys,inspect
from termcolor import colored
import torch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import models as Models
import global_vars as Global
from utils.args import args

import categories.classifier_setup as CLSetup
from models.classifiers import NIHDenseBinary
from datasets.NIH_Chest import NIHChestBinaryTrainSplit

if __name__ == "__main__":
    dataset = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, "NIHCC"), binary=True)
    model = NIHDenseBinary("mono_model.pth.tar")
    CLSetup.train_classifier(args, model=model, dataset=dataset.get_D1_train())

    # task_list = [
    #     # The list of models,   The function that does the training,    Can I skip-test?,   suffix of the operation.
    #     # The procedures that can be skip-test are the ones that we can determine
    #     # whether we have done them before without instantiating the network architecture or dataset.
    #     # saves quite a lot of time when possible.
    #     (Global.dataset_reference_classifiers, CLSetup.train_classifier,            True, ['base0']),
    #     (Global.dataset_reference_classifiers, KLogisticSetup.train_classifier,     True, ['KLogistic']),
    #     (Global.dataset_reference_classifiers, DeepEnsembleSetup.train_classifier,  True, ['DE.%d'%i for i in range(5)]),
    #     (Global.dataset_reference_autoencoders, AESetup.train_BCE_AE,               False, []),
    #     (Global.dataset_reference_autoencoders, AESetup.train_MSE_AE,               False, []),
    #     (Global.dataset_reference_vaes, AESetup.train_variational_autoencoder,      False, []),
    #     (Global.dataset_reference_pcnns, PCNNSetup.train_pixelcnn,                  False, []),
    # ]
    #
    # # Do a for loop to run the training tasks.
    # for task_id, (ref_list, train_func, skippable, suffix) in enumerate(task_list):
    #     target_datasets = ref_list.keys()
    #     print('Processing %d datasets.'%len(target_datasets))
    #     for dataset in target_datasets:
    #         print('Processing dataset %s with %d networks for %d-%s.'%(colored(dataset, 'green'), len(ref_list[dataset]), task_id, colored(train_func.__name__, 'blue')))
    #         if skippable and not needs_processing(args, Global.all_datasets[dataset], ref_list[dataset], suffix=suffix):
    #             print(colored('Skipped', 'yellow'))
    #             continue
    #         ds = Global.all_datasets[dataset]()
    #         for model in ref_list[dataset]:
    #             model = model()
    #             print('Training %s'%(colored(model.__class__.__name__, 'blue')))
    #             train_func(args, model, ds.get_D1_train())
