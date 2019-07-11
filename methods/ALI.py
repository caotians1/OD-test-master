from __future__ import print_function
from os import path
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim

from torch.utils.data import DataLoader
from utils.iterative_trainer import IterativeTrainerConfig, IterativeTrainer
from utils.logger import Logger

from methods import AbstractMethodInterface, AbstractModelWrapper, SVMLoss
from methods.base_threshold import ProbabilityThreshold
from methods.reconstruction_error import ReconstructionThreshold
from datasets import MirroredDataset
import global_vars as Global
from models.ALImodel import ALIModel


class ALIReconstruction(ReconstructionThreshold):
    def method_identifier(self):
        output = "ALIReconstruction"
        if len(self.add_identifier) > 0:
            output = output + "/" + self.add_identifier
        return output

    def get_base_config(self, dataset):
        print("Preparing training D1 for %s" % (dataset.name))

        # Initialize the multi-threaded loaders.
        all_loader = DataLoader(dataset, batch_size=self.args.batch_size, num_workers=self.args.workers,
                                pin_memory=True)

        # Set up the model
        model = Global.get_ref_ali(dataset.name)[0]().to(self.args.device)

        # Set up the criterion
        criterion = None
        if self.default_model == 0:
            criterion = nn.BCEWithLogitsLoss().to(self.args.device)
        else:
            criterion = nn.MSELoss().to(self.args.device)
            model.default_sigmoid = True

        # Set up the config
        config = IterativeTrainerConfig()

        config.name = '%s-ALIAE1' % (self.args.D1)
        config.phases = {
            'all': {'dataset': all_loader, 'backward': False},
        }
        config.criterion = criterion
        config.classification = False
        config.cast_float_label = False
        config.autoencoder_target = True
        config.stochastic_gradient = True
        config.visualize = not self.args.no_visualize
        config.sigmoid_viz = self.default_model == 0
        config.model = model
        config.optim = None
        config.logger = Logger()

        return config

    def propose_H(self, dataset):
        config = self.get_base_config(dataset)

        import models as Models

        if self.default_model == 0:
            config.model.netid = "BCE." + config.model.netid
        else:
            config.model.netid = "MSE." + config.model.netid


        home_path = Models.get_ref_model_path(self.args, config.model.__class__.__name__, dataset.name,
                                              suffix_str=config.model.netid)
        hbest_path = path.join(home_path, 'model.best.pth')
        best_h_path = hbest_path

        # trainer = IterativeTrainer(config, self.args)

        if not path.isfile(best_h_path):
            raise NotImplementedError(
                "%s not found!, Please use setup_model to pretrain the networks first!" % best_h_path)
        else:
            print(colored('Loading H1 model from %s' % best_h_path, 'red'))
            config.model.load_state_dict(torch.load(best_h_path))

        # trainer.run_epoch(0, phase='all')
        # test_loss = config.logger.get_measure('all_loss').mean_epoch(epoch=0)
        # print("All average loss %s"%colored('%.4f'%(test_loss), 'red'))

        self.base_model = config.model
        self.base_model.eval()