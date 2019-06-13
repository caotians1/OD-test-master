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

class NIHChestBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir, split, index_file="Data_Entry_2017.csv", image_dir="images", imsize=256):
        super(NIHChestBase,self).__init__()
        self.index_cache_path = index_cache_path
        self.source_dir = source_dir
        self.split = split
        self.cache_file = "NIH_Proc.pkl"
        self.index_file = index_file
        self.image_dir = image_dir
        self.imsize = imsize
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.Resize((self.imsize, self.imsize)),
                                              transforms.ToTensor()])
        assert split in ["train", "val", "test"]

        if not osp.exists(osp.join(self.index_cache_path, self.cache_file)):
            self.generate_index()
        cache_file = torch.load(osp.join(self.index_cache_path, self.cache_file))
        self.img_list = cache_file['img_list']
        self.label_tensors = cache_file['label_tensors']

        if not (osp.exists(osp.join(self.source_dir, 'val_split.pt'))
                and osp.exists(osp.join(self.source_dir, 'train_split.pt'))
                and osp.exists(osp.join(self.source_dir, 'test_split.pt'))):
            self.generate_split()

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
        missing_train = 0
        missing_val = 0
        missing_test = 0
        for entry in lines[:train_num]:
            try:
                train_inds.append(self.img_list.index(entry.strip("\n")))
            except ValueError:
                missing_train += 1
        for entry in lines[train_num:]:
            try:
                val_inds.append(self.img_list.index(entry.strip("\n")))
            except ValueError:
                missing_val += 1

        with open(osp.join(self.source_dir, 'test_list.txt'), 'r+') as fp:
            lines = fp.readlines()
        for entry in lines:
            try:
                test_inds.append(self.img_list.index(entry.strip("\n")))
            except ValueError:
                missing_test += 1
        print("%i, %i,and %i found in original split file; %i, %i, and %i missing" %
              (train_num, n_trainval - train_num, len(lines), missing_train, missing_val, missing_test))

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
        with open(osp.join(self.source_dir, self.index_file), 'r') as fp:
            csvf = csv.DictReader(fp)
            for row in csvf:
                imp = osp.join(self.source_dir, self.image_dir, row['Image Index'])
                if osp.exists(imp):
                    img_list.append(row['Image Index'])
                    findings = row['Finding Labels'].split('|')
                    label = [1 if cond in findings else 0 for cond in CLASSES]
                    label_list.append(label)
        label_tensors = torch.IntTensor(label_list)
        os.makedirs(self.index_cache_path, exist_ok=True)
        torch.save({'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list},
                   osp.join(self.index_cache_path, self.cache_file))
        return

class NIHChest(AbstractDomainInterface):
    """
    Wrapper for using all classes of NIHChest as train or test dataset
    """
    def __init__(self):
        super(NIHChest, self).__init__()
        cache_path = "E:\ChestXray-NIHCC"
        source_path = "E:\ChestXray-NIHCC"
        self.ds_train = NIHChestBase(cache_path, source_path, "train")
        self.ds_valid = NIHChestBase(cache_path, source_path, "val")
        self.ds_test = NIHChestBase(cache_path, source_path, "test")
        train_indices = torch.randperm(len(self.ds_train))
        self.D1_train_ind = train_indices.int()
        self.D1_valid_ind = torch.arange(0, len(self.ds_valid)).int()
        self.D1_test_ind = torch.arange(0, len(self.ds_test)).int()

        self.D2_valid_ind = train_indices.int()
        self.D2_test_ind = torch.arange(0, len(self.ds_valid)).int()
        self.image_size = (256, 256)

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
        return SubDataset(self.name, self.ds_valid, target_indices, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        return transforms.Compose([transforms.ToPILImage(),
                                   transforms.Grayscale(),
                                   transforms.Resize((256, 256)),
                                   transforms.ToTensor()
                                   ])

if __name__ == "__main__":
    dataset = NIHChest()
    d1_train = dataset.get_D1_train()
    print(len(d1_train))
    loader = data.DataLoader(d1_train, batch_size=1, shuffle=True)
    import matplotlib.pyplot as plt
    for batch, batch_ind in zip(loader, range(10)):
        print(batch_ind)
        x, y = batch
        plt.imshow(x.numpy().reshape(dataset.image_size))

    d2_valid = dataset.get_D2_valid(dataset)
    print(len(d2_valid))
    loader = data.DataLoader(d2_valid, batch_size=1, shuffle=True)
    import matplotlib.pyplot as plt

    for batch, batch_ind in zip(loader, range(10)):
        print(batch_ind)
        x, y = batch
        plt.imshow(x.numpy().reshape(dataset.image_size))
