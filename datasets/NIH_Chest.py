import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface
import os
import os.path as osp
import csv
import subprocess
from PIL import Image

CLASSES = ['Effusion', 'Emphysema', 'Pneumonia', 'Cardiomegaly', 'Pneumothorax', 'Mass', 'Infiltration', 'No_Finding',
           'Nodule', 'Consolidation', 'Atelectasis', 'Edema', 'Fibrosis', 'Hernia', 'Pleural_Thickening']
N_CLASS = len(CLASSES)

class NIH_ChestBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir, split, index_file="Data_Entry_2017.csv", image_dir="images", imsize=256):
        super(NIH_ChestBase,self).__init__()
        self.index_cache_path = index_cache_path
        self.source_dir = source_dir
        self.split = split
        self.cache_file = "NIH_Proc.pkl"
        self.index_file = index_file
        self.image_dir = image_dir
        self.imsize = imsize
        self.transforms = transforms.Compose([transforms.Grayscale,
                                              transforms.Resize((self.imsize, self.imsize)),
                                              transforms.ToTensor()])
        assert split in ["train", "val", "test"]

        if not osp.exists(osp.join(self.index_cache_path, self.cache_file)):
            self.generate_index()

        if not (osp.exists(osp.join(self.source_dir, 'val_split.pt'))
                and osp.exists(osp.join(self.source_dir, 'train_split.pt'))
                and osp.exists(osp.join(self.source_dir, 'test_split.pt'))):
            self.generate_split()
        cache_file = torch.load(osp.join(self.index_cache_path, self.cache_file))
        self.img_list = cache_file['img_list']
        self.label_tensors = cache_file['label_tensors']
        self.split_inds = torch.load(osp.join(self.index_cache_path, "%s_split.pt"% self.split))

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, item):
        index = self.split_inds[item]
        img_name = self.img_list[index]
        label = self.label_tensors[index]
        imp = osp.join(self.source_dir, self.image_dir, img_name)
        with open(imp, 'rb') as f:
            with Image.open(f) as img:
                img = self.transforms(img)
        return img, label

    def generate_split(self):
        with open(osp.join(self.source_dir, 'train_val_list.txt'), 'r+') as fp:
            lines = fp.readlines()

        n_trainval = len(lines)
        train_num = int(n_trainval * 7.0 / 8.0)
        train_inds = []
        val_inds = []
        test_inds = []
        for entry in lines[:train_num]:
            train_inds.append(self.img_list.index(entry))
        for entry in lines[train_num:]:
            val_inds.append(self.img_list.index(entry))

        with open(osp.join(self.source_dir, 'test_list.txt'), 'r+') as fp:
            lines = fp.readlines()
        for entry in lines:
            test_inds.append(self.img_list.index(entry))

        torch.save(train_inds, osp.join(self.index_cache_path, "train_split.pt"))
        torch.save(val_inds, osp.join(self.index_cache_path, "val_split.pt"))
        torch.save(test_inds, osp.join(self.index_cache_path, "test_split.pt"))
        return

    def generate_index(self):
        """
        Scan index file to create list of images and labels for each image. Also stores index files in index_cache_path
        :return:
        """
        img_list = []
        label_list = []
        with open(osp.join(self.source_dir, self.index_file), 'r+') as fp:
            csvf = csv.DictReader(fp)
            for row in csvf:
                imp = osp.join(self.source_dir, self.image_dir, row['Image_Index'])
                if osp.exists(imp):
                    img_list.append(row['Image_Index'])
                    findings = row['Finding Labels'].split('|')
                    label = [1 if cond in findings else 0 for cond in CLASSES]
                    label_list.append(label)
        label_tensors = torch.IntTensor(label_list)
        os.makedirs(self.index_cache_path, exist_ok=True)
        torch.save({'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list},
                   osp.join(self.index_cache_path, self.cache_file))
        return

class NIH_Chest(AbstractDomainInterface):
    """
    based on requirements, data loader need "leave-one-out" function
    """

    def __init__(self):
        super(NIH_Chest, self).__init__()

        im_transformer = transforms.Compose([transforms.ToTensor()])
        root_path = './workspace/datasets/mnist'
        self.D1_train_ind = torch.arange(0, 50000).int()
        self.D1_valid_ind = torch.arange(50000, 60000).int()
        self.D1_test_ind = torch.arange(0, 10000).int()

        self.D2_valid_ind = torch.arange(0, 60000).int()
        self.D2_test_ind = torch.arange(0, 10000).int()

        self.ds_train = datasets.MNIST(root_path,
                                       train=True,
                                       transform=im_transformer,
                                       download=True)
        self.ds_test = datasets.MNIST(root_path,
                                      train=False,
                                      transform=im_transformer,
                                      download=True)

    def get_D1_train(self):
        return SubDataset(self.name, self.ds_train, self.D1_train_ind)

    def get_D1_valid(self):
        return SubDataset(self.name, self.ds_train, self.D1_valid_ind, label=0)

    def get_D1_test(self):
        return SubDataset(self.name, self.ds_test, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_train, self.D2_valid_ind, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        return SubDataset(self.name, self.ds_test, self.D2_test_ind, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((28, 28)),
                                   transforms.Grayscale(),
                                   transforms.ToTensor()
                                   ])