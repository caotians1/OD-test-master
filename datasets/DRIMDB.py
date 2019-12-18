import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
import os
import os.path as osp
import csv
import subprocess
from PIL import Image
import numpy as np


class DRIMDBBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir, split, image_path="images_224.npy", imsize=224, transforms=None,
                 to_gray=False, download=False, extract=True):
        super(DRIMDBBase,self).__init__()
        self.index_cache_path = index_cache_path
        self.source_dir = source_dir
        self.split = split
        self.imsize = imsize
        self.image_path = image_path
        self.to_gray = to_gray
        if transforms is None:
            self.transforms = transforms.Compose([transforms.Resize((imsize, imsize)),
                                                  transforms.ToTensor()])
        else:
            self.transforms = transforms
        assert split in ["train", "val", "test"]
        if extract:
            self.data = np.load(osp.join(source_dir, image_path))
            self.img_list = np.arange(len(self.data))
            if not (osp.exists(osp.join(self.source_dir, 'val_split.pt'))
                    and osp.exists(osp.join(self.source_dir, 'train_split.pt'))
                    and osp.exists(osp.join(self.source_dir, 'test_split.pt'))):
                self.generate_split()

            self.split_inds = torch.load(osp.join(self.index_cache_path, "%s_split.pt"% self.split))

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, item):
        index = self.split_inds[item]
        img = self.data[index]
        img = Image.fromarray(img)
        if not self.to_gray:
            img = self.transforms(img.convert('RGB'))
        else:
            img = self.transforms(img.convert('L'))
        return img, torch.LongTensor([0,])

    def generate_split(self):
        n_total = len(self.img_list)
        train_num = int(0.6*n_total)
        val_num = int(0.7*n_total)
        train_inds = np.arange(train_num)
        val_inds = np.arange(start=train_num, stop=val_num)
        test_inds = np.arange(start=val_num, stop=n_total)

        torch.save(train_inds, osp.join(self.index_cache_path, "train_split.pt"))
        torch.save(val_inds, osp.join(self.index_cache_path, "val_split.pt"))
        torch.save(test_inds, osp.join(self.index_cache_path, "test_split.pt"))
        return

class DRIMDB(AbstractDomainInterface):

    dataset_path = "DRIMDB"
    def __init__(self, root_path="./workspace/datasets/DRIMDB", downsample=None, shrink_channels=False, test_length=None, download=False,
                 extract=True, doubledownsample=None):
        """
        :param leave_out_classes: if a sample has ANY class from this list as positive, then it is removed from indices.
        :param keep_in_classes: when specified, if a sample has None of the class from this list as positive, then it
         is removed from indices..
        """
        self.name = "DRIMDB"
        super(DRIMDB, self).__init__()
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
                                            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            #                     std=[0.229, 0.224, 0.225]),
                                            ])
            self.image_size = (downsample, downsample)
        else:
            transform = transforms.Compose(transform_list +
                                            [transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                             #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             #                     std=[0.229, 0.224, 0.225]),
                                             ])
            self.image_size = (224, 224)

        self.ds_train = DRIMDBBase(cache_path, source_path, "train", transforms=transform,
                                     to_gray=shrink_channels, download=download, extract=extract)
        self.ds_valid = DRIMDBBase(cache_path, source_path, "val", transforms=transform,
                                to_gray=shrink_channels, download=download, extract=extract)
        self.ds_test = DRIMDBBase(cache_path, source_path, "test", transforms=transform,
                               to_gray=shrink_channels, download=download, extract=extract)
        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True, max_l=self.max_l)
            self.D1_test_ind = self.get_filtered_inds(self.ds_test, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_test)


    def get_filtered_inds(self, basedata: DRIMDBBase, shuffle=False, max_l=None):
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
        target = 224
        if self.downsample is not None:
            target = self.downsample
        if self.shrink_channels:
            return transforms.Compose([ExpandRGBChannels(),
                                       transforms.ToPILImage(),
                                       transforms.Grayscale(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor()
                                       ])
        else:
            return transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor(),
                                       transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                       ])

if __name__ == "__main__":
    dataset = DRIMDB()
    d1_train = dataset.get_D1_train()
    print(len(d1_train))
    loader = data.DataLoader(d1_train, batch_size=1, shuffle=True)
    import matplotlib.pyplot as plt
    for batch, batch_ind in zip(loader, range(10)):
        print(batch_ind)
        x, y = batch
        plt.imshow(x[0].numpy().transpose((1,2,0)))
