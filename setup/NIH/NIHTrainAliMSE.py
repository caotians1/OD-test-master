from __future__ import print_function
import os,sys,inspect
from termcolor import colored
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)


from utils.args import args
from utils.logger import Logger

import setup.categories.ali_setup as ALISetup
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler

from models.ALImodel import ALIModel
from datasets.NIH_Chest import NIHChestBinaryTrainSplit


from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = NIHChestBinaryTrainSplit(root_path=os.path.join(args.root_path, "NIHCC"), binary=True,
                                       expand_channels=False, downsample=64).get_D1_train()
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, True, num_workers=args.workers, pin_memory=True)
    model = ALIModel(dims=(1,64,64)).cuda()
    ALISetup.Train_ALI(args, model, dataset, BCE_Loss=True)

