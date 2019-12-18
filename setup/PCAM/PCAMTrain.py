from __future__ import print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from utils.args import args
import setup.categories.classifier_setup as CLSetup
from models.classifiers import PCAMDense
from datasets.PCAM import PCAM


if __name__ == "__main__":
    dataset = PCAM(root_path=os.path.join(args.root_path, "pcam"), extract=True, downsample=224)
    model = PCAMDense("densenet121-a639ec97.pth", train_features=True)
    CLSetup.train_classifier(args, model=model, dataset=dataset.get_D1_train(), balanced=True)

