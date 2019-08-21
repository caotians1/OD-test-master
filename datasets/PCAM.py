import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
import os
import os.path as osp
import csv
import numpy as np
import subprocess
from PIL import Image
from zipfile import ZipFile
import gzip
import h5py

class PCAMBase(data.Dataset):
    def __init__(self, source_dir, split, imsize=96, transforms=None,
                 to_gray=False, download=False, extract=True):
        super(PCAMBase,self).__init__()
        self.source_dir = source_dir
        self.split = split
        self.imsize = imsize
        self.to_gray = to_gray
        if transforms is None:
            self.transforms = transforms.ToTensor()
        else:
            self.transforms = transforms
        assert split in ["train", "valid", "test"]
        if extract:
            self.extract()
            self.h5x = h5py.File(osp.join(self.source_dir, "camelyonpatch_level_2_split_%s_x.h5" % self.split), mode='r', swmr=True)
            self.h5y = h5py.File(osp.join(self.source_dir, "camelyonpatch_level_2_split_%s_y.h5" % self.split), mode='r', swmr=True)
            self.label_tensors = self.h5y['y']

    def __len__(self):
        return len(self.label_tensors)

    def __getitem__(self, item):
        x = self.h5x['x'][item]
        label = torch.LongTensor(self.label_tensors[item]).squeeze()
        if self.to_gray:
            img = self.transforms(transforms.ToPILImage()(x).convert('L'))
        else:
            img = self.transforms(transforms.ToPILImage()(x).convert('RGB'))
        return img, label

    def extract(self):
        if os.path.exists(os.path.join(self.source_dir, "camelyonpatch_level_2_split_test_x.h5")):
            return
        import shutil
        tarsplits_list = ["camelyonpatch_level_2_split_test_x.h5.gz",
                          "camelyonpatch_level_2_split_train_x.h5.gz",
                          "camelyonpatch_level_2_split_valid_x.h5.gz",
                          "camelyonpatch_level_2_split_train_y.h5.gz",
                          "camelyonpatch_level_2_split_valid_y.h5.gz",
                          "camelyonpatch_level_2_split_test_y.h5.gz",
                     ]
        for tar_split in tarsplits_list:
            with gzip.open(os.path.join(self.source_dir, tar_split), 'rb') as f_in:
                with open(os.path.join(self.source_dir, tar_split.split(".")[0]), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


class PCAM(AbstractDomainInterface):

    dataset_path = "pcam"
    def __init__(self, root_path="./workspace/datasets/pcam", downsample=None, shrink_channels=False, test_length=None, download=False,
                 extract=True, doubledownsample=None):
        """
        :param leave_out_classes: if a sample has ANY class from this list as positive, then it is removed from indices.
        :param keep_in_classes: when specified, if a sample has None of the class from this list as positive, then it
         is removed from indices..
        """
        self.name = "PCAM"
        super(PCAM, self).__init__()
        self.downsample = downsample
        self.shrink_channels=shrink_channels
        self.max_l = test_length
        cache_path = root_path
        source_path = root_path
        if doubledownsample is not None:
            transform_list = [transforms.Resize(doubledownsample),]
        else:
            transform_list = []
        if downsample is not None:
            print("downsampling to", downsample)
            transform = transforms.Compose(transform_list +
                                           [transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                            ])
            self.image_size = (downsample, downsample)
        else:
            transform = transforms.Compose(transform_list + [transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225]),
                                                             ])
            self.image_size = (224, 224)

        self.ds_train = PCAMBase(source_path, "train", imsize=self.image_size[0], transforms=transform,
                                     to_gray=shrink_channels, download=download, extract=extract)
        self.ds_valid = PCAMBase(source_path, "valid", imsize=self.image_size[0], transforms=transform,
                                to_gray=shrink_channels, download=download, extract=extract)
        self.ds_test = PCAMBase(source_path, "test", imsize=self.image_size[0], transforms=transform,
                               to_gray=shrink_channels, download=download, extract=extract)
        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True, max_l=self.max_l)
            self.D1_test_ind = self.get_filtered_inds(self.ds_test, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_test)


    def get_filtered_inds(self, basedata: PCAMBase, shuffle=False, max_l=None):
        output_inds = torch.arange(0, len(basedata)).int()
        if shuffle:
            output_inds = output_inds[torch.randperm(len(output_inds))]
        if max_l is not None:
            if len(output_inds) >max_l:
                output_inds = output_inds[:max_l]
        return output_inds

    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)

    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_valid, self.D1_valid_ind, label=0)

    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_valid_ind
        return SubDataset(self.name, self.ds_train, target_indices, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_test_ind
        return SubDataset(self.name, self.ds_test, target_indices, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        target = self.image_size[0]
        if self.shrink_channels:
            return transforms.Compose([ExpandRGBChannels(),
                                       transforms.ToPILImage(),
                                       transforms.Grayscale(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor()
                                       ])
        else:
            return transforms.Compose([
                                       ExpandRGBChannels(),
                                       transforms.ToPILImage(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor(),
                                       ])
