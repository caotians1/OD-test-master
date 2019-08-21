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
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from torch import optim
import categories.classifier_setup as CLSetup
from models.ALImodel import ALIModel
from datasets.PCAM import PCAM
from utils.logger import Logger
from tqdm import tqdm

if __name__ == "__main__":
    dataset = PCAM(root_path=os.path.join(args.root_path, "pcam"), extract=True, downsample=64).get_D1_train()
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, True, num_workers=args.workers, pin_memory=True)
    model = ALIModel(dims=(3,64,64)).cuda()
    logger = Logger()
    home_path = Models.get_ref_model_path(args, model.__class__.__name__, dataset.name, model_setup=True,
                                          suffix_str='base0')
    hbest_path = os.path.join(home_path, 'model.best.pth')

    if not os.path.isdir(home_path):
        os.makedirs(home_path)
    best_gen_loss = 9999
    if not os.path.isfile(hbest_path + ".done"):
        print(colored('Training from scratch', 'green'))
        best_loss = -1

        optimizerG = optim.Adam([{'params': model.GenX.parameters()},
                                 {'params': model.GenZ.parameters()}], lr=args.lr, betas=(args.beta1, args.beta2))

        optimizerD = optim.Adam([{'params': model.DisZ.parameters()}, {'params': model.DisX.parameters()},
                                 {'params': model.DisXZ.parameters()}], lr=args.lr, betas=(args.beta1, args.beta2))
        criterion = nn.BCELoss()
        for epoch in range(1, 100 + 1):
            model.train()
            with tqdm(total=len(dataloader), disable=bool(os.environ.get("DISABLE_TQDM", False))) as pbar:
                for i, (x, y) in enumerate(dataloader):
                    pbar.update()
                    batchsize = x.shape[0]
                    fakeZ = torch.randn(batchsize, 512, 1, 1).cuda()
                    pred_real, pred_fake = model.forward(x.cuda(), fakeZ)
                    truelabel = torch.ones(batchsize) - 0.1
                    fakelabel = torch.zeros(batchsize)

                    if args.random_label == True:
                        truelabel = torch.randint(low=70, high=110, size=(1, batchsize))[0] / 100
                        fakelabel = torch.randint(low=-10, high=30, size=(1, batchsize))[0] / 100
                    truelabel = truelabel.cuda()
                    fakelabel = fakelabel.cuda()
                    loss_d = criterion(pred_real.view(-1), truelabel) + criterion(pred_fake.view(-1), fakelabel)
                    loss_g = criterion(pred_fake.view(-1), truelabel) + criterion(pred_real.view(-1), fakelabel)
                    logger.log('Disc_loss', loss_d.item(), epoch, i)
                    logger.log('Gen_loss', loss_g.item(), epoch, i)


                    if loss_g > args.max_loss_g:
                        optimizerG.zero_grad()
                        loss_g.backward()
                        optimizerG.step()
                        pbar.set_description("Skipped D, Disc_loss %.4f, Gen_loss %.4f" % (loss_d.item(), loss_g.item()))
                    elif loss_g < args.min_loss_g:
                        optimizerD.zero_grad()
                        loss_d.backward()
                        optimizerD.step()
                        pbar.set_description(
                            "Skipped G, Disc_loss %.4f, Gen_loss %.4f" % (loss_d.item(), loss_g.item()))
                    else:
                        optimizerD.zero_grad()
                        loss_d.backward(retain_graph=True)
                        optimizerD.step()
                        optimizerG.zero_grad()
                        loss_g.backward()
                        optimizerG.step()
                        pbar.set_description(
                            "Disc_loss %.4f, Gen_loss %.4f" % (loss_d.item(), loss_g.item()))

            disc_loss = logger.get_measure('Disc_loss').mean_epoch()
            gen_loss = logger.get_measure('Gen_loss').mean_epoch()
            print("Discriminator loss %.4f, Generator loss %.4f" % (disc_loss, gen_loss))

            torch.save(logger.measures, os.path.join(home_path, 'logger.pth'))

            if args.save and gen_loss < best_gen_loss:
                print('Updating the on file model with %s' % (colored('%.4f' % gen_loss, 'red')))
                best_gen_loss = gen_loss
                torch.save(model.state_dict(), hbest_path)
