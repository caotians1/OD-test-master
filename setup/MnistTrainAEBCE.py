from __future__ import print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from utils.args import args

import categories.ae_setup as AESetup
from models.autoencoders import *
from datasets.MNIST import MNIST

if __name__ == "__main__":
    dataset = MNIST(root_path=os.path.join(args.root_path, "mnist"), download=True, extract=True)
    model = Generic_AE(dims=(1, 28, 28), max_channels=128, depth=6, n_hidden=128)
    AESetup.train_autoencoder(args, model=model, dataset=dataset.get_D1_train(), BCE_Loss=True)

