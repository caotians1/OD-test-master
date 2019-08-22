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
import torchvision
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from torch import optim
import categories.classifier_setup as CLSetup
from models.classifiers import DRDDense
from datasets.DRD import DRD

class DRDDenseCustom(DRDDense):
    def train_config(self):
        config = {}
        if self.train_features:
            config['optim'] = optim.Adam(
                [{'params':self.densenet121.classifier.parameters(), 'lr':1e-1}, {'params':self.densenet121.features.parameters()}],
                lr=1e-1)
        else:
            config['optim'] = optim.Adam(self.densenet121.classifier.parameters(), lr=1e-3, )
        config['scheduler'] = optim.lr_scheduler.StepLR(config['optim'], 30, gamma=0.1)
        config['max_epoch'] = 900
        return config

if __name__ == "__main__":
    dataset = DRD(root_path=os.path.join(args.root_path, "diabetic-retinopathy-detection"))
    model = DRDDense("densenet121-a639ec97.pth", train_features=True)
    args.num_classes = 2
    CLSetup.train_classifier(args, model=model, dataset=dataset.get_D1_train(), balanced=True)

