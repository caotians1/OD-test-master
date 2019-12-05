import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
import os
import os.path as osp
import csv
import subprocess
import pickle
from PIL import Image
import numpy as np

CLASSES = ["breast_ER_patches", "breast_HE_patches", "kidney_HE_patches", "kidney_MAS_patches"]

class ANHIRBase(data.Dataset):
    def __init__(self, source_dir, split, image_path="images_96.npy", label_path="labels.pkl", imsize=224, transforms=None,
                 to_gray=False, download=False, extract=True):
        super(ANHIRBase,self).__init__()
        self.index_cache_path = source_dir
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
        assert split in ["train", "valid", "test"]
        if extract:
            self.data = np.load(osp.join(source_dir, image_path))
            self.img_list = np.arange(len(self.data))
            with open(osp.join(source_dir, label_path), "rb") as fp:
                str_labels = pickle.load(fp)
                numeric_labels = []
                for l in str_labels:
                    label = np.zeros(5, dtype=np.int64)
                    label[CLASSES.index(l)] = 1
                    numeric_labels.append(label)
                labels = np.stack(numeric_labels)
                self.labels = torch.LongTensor(labels)

            if not (osp.exists(osp.join(self.source_dir, 'valid_split.pt'))
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
        return img, self.labels[index]

    def generate_split(self):
        n_total = len(self.img_list)
        train_num = int(0.7*n_total)
        val_num = int(0.8*n_total)
        train_inds = np.arange(train_num)
        val_inds = np.arange(start=train_num, stop=val_num)
        test_inds = np.arange(start=val_num, stop=n_total)

        torch.save(train_inds, osp.join(self.index_cache_path, "train_split.pt"))
        torch.save(val_inds, osp.join(self.index_cache_path, "valid_split.pt"))
        torch.save(test_inds, osp.join(self.index_cache_path, "test_split.pt"))
        return


class ANHIR(AbstractDomainInterface):
    dataset_path = "ANHIR"
    def __init__(self, root_path="./workspace/datasets/ANHIR", downsample=None, shrink_channels=False, test_length=None, download=False,
                 extract=True, doubledownsample=None):
        """
        :param leave_out_classes: if a sample has ANY class from this list as positive, then it is removed from indices.
        :param keep_in_classes: when specified, if a sample has None of the class from this list as positive, then it
         is removed from indices..
        """
        self.name = "ANHIR"
        super(ANHIR, self).__init__()
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
            transform_list += [transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor(),]
            if self.shrink_channels:
                transform_list += [transforms.Grayscale(),]
            else:
                transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
            transform = transforms.Compose(transform_list)
            self.image_size = (downsample, downsample)
        else:
            transform_list += [transforms.Resize((224, 224)),
                               transforms.ToTensor(), ]
            if self.shrink_channels:
                transform_list += [transforms.Grayscale(),]
            else:
                transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
            transform = transforms.Compose(transform_list)
            self.image_size = (224, 224)

        self.ds_train = ANHIRBase(source_path, "train", imsize=self.image_size[0], transforms=transform,
                                     to_gray=shrink_channels, download=download, extract=extract)
        self.ds_valid = ANHIRBase(source_path, "valid", imsize=self.image_size[0], transforms=transform,
                                to_gray=shrink_channels, download=download, extract=extract)
        self.ds_test = ANHIRBase(source_path, "test", imsize=self.image_size[0], transforms=transform,
                               to_gray=shrink_channels, download=download, extract=extract)
        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True, max_l=self.max_l)
            self.D1_test_ind = self.get_filtered_inds(self.ds_test, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_test)


    def get_filtered_inds(self, basedata: ANHIRBase, shuffle=False, max_l=None):
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

if __name__ == "__main__":
    data1 = ANHIR("workspace\\datasets\\ANHIR")
    d1 = data1.get_D1_train()
    print(len(d1))
    for i in range(10):
        x, y = d1[i]