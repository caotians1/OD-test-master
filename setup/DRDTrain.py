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
from models.classifiers import DRDDense
from datasets.DRD import DRD

if __name__ == "__main__":
    dataset = DRD(root_path=os.path.join(args.root_path, "diabetic-retinopathy-detection"))
    model = DRDDense("densenet121-a639ec97.pth")
    CLSetup.train_classifier(args, model=model, dataset=dataset.get_D1_train())

