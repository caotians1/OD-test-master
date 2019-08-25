from __future__ import print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from utils.args import args
import setup.categories.classifier_setup as CLSetup
from models.classifiers import DRDDense
from datasets.DRD import DRD
from torch import optim

class DRDDenseCustom(DRDDense):
    def train_config(self):
        config = {}
        if self.train_features:
            config['optim'] = optim.Adam(
                [{'params':self.densenet121.classifier.parameters(), 'lr':1e-3}, {'params':self.densenet121.features.parameters()}],
                lr=1e-3)
        else:
            config['optim'] = optim.Adam(self.densenet121.classifier.parameters(), lr=1e-3, )
        config['scheduler'] = optim.lr_scheduler.StepLR(config['optim'], 30, gamma=0.5)
        config['max_epoch'] = 300
        return config

if __name__ == "__main__":
    dataset = DRD(root_path=os.path.join(args.root_path, "diabetic-retinopathy-detection"))
    model = DRDDenseCustom(train_features=True)
    args.num_classes = 2
    CLSetup.train_classifier(args, model=model, dataset=dataset.get_D1_train(), balanced=True)

