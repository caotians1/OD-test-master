from __future__ import print_function
import os,sys,inspect
from termcolor import colored
import torch
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

import models as Models
import global_vars as Global
from utils.args import args

import setup.categories.classifier_setup as CLSetup
from models.classifiers import PADDense
from datasets.PADChest import PADChestBinaryTrainSplit

if __name__ == "__main__":
    dataset = PADChestBinaryTrainSplit(root_path=os.path.join(args.root_path, "PADChest"), binary=True, downsample=224)
    model = PADDense("densenet121-a639ec97.pth", train_features=True)
    args.num_classes = 2
    CLSetup.train_classifier(args, model=model, dataset=dataset.get_D1_train(), balanced=True)

