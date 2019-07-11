from __future__ import print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utils.args import args

import categories.ae_setup as AESetup
from models.autoencoders import Residual_AE
from datasets.MNIST import MNIST

if __name__ == "__main__":
    dataset = MNIST(root_path=currentdir,download=True, extract=True)
    model = Residual_AE(dims=(1, 32, 32), max_channels=256, depth=5, n_hidden=256)
    AESetup.train_autoencoder(args, model=model, dataset=dataset.get_D1_train(), BCE_Loss=False)

