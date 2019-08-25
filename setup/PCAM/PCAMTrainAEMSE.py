from __future__ import print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from utils.args import args

import setup.categories.ae_setup as AESetup
from models.autoencoders import *
from datasets.PCAM import PCAM

if __name__ == "__main__":
    dataset = PCAM(root_path=os.path.join(args.root_path, "pcam"), extract=True, downsample=64)
    model = Generic_AE(dims=(3, 64, 64), max_channels=512, depth=12, n_hidden=512)
    AESetup.train_autoencoder(args, model=model, dataset=dataset.get_D1_train(), BCE_Loss=False)

