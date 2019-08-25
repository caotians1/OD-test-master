from __future__ import print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from utils.args import args

import setup.categories.ae_setup as AESetup
from models.autoencoders import *
from datasets.DRD import DRD

if __name__ == "__main__":
    dataset = DRD(root_path=os.path.join(args.root_path, "diabetic-retinopathy-detection"), downsample=64)
    model = ALILikeVAE(dims=(3, 64, 64))
    AESetup.train_variational_autoencoder(args, model=model, dataset=dataset.get_D1_train(), BCE_Loss=False)

