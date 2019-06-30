import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
import os
import os.path as osp
import csv
import subprocess
from PIL import Image

CLASSES = ['Effusion', 'Emphysema', 'Pneumonia', 'Cardiomegaly', 'Pneumothorax', 'Mass', 'Infiltration', 'No Finding',
           'Nodule', 'Consolidation', 'Atelectasis', 'Edema', 'Fibrosis', 'Hernia', 'Pleural_Thickening']
N_CLASS = len(CLASSES)
MAX_LENGTH = 1000000
def to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def group_normalize(crops):
    return torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])(crop) for crop in crops])

class NIHChestBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir, split, index_file="Data_Entry_2017.csv", image_dir="images",
                 imsize=224, transforms=None, binary=False, download=False, extract=True):
        super(NIHChestBase,self).__init__()
        self.index_cache_path = index_cache_path
        self.source_dir = source_dir
        self.split = split
        self.cache_file = "NIHChestIndex.pkl"
        self.index_file = index_file
        self.image_dir = image_dir
        self.imsize = imsize
        self.binary = binary
        if transforms is None:
            self.transforms = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.RandomCrop((imsize,imsize)),
                                              transforms.ToTensor()])
        else:
            self.transforms = transforms
        assert split in ["train", "val", "test"]
        if extract:
            self.extract()
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
        if self.binary:
            label = label[7]
        imp = osp.join(self.source_dir, self.image_dir, img_name)
        with open(imp, 'rb') as f:
            with Image.open(f).convert('RGB') as img:
                img = self.transforms(img)
        return img, label

    def extract(self):
        if os.path.exists(os.path.join(self.source_dir, self.image_dir)):
            return
        import tarfile
        tarsplits_list = ["images_01.tar.gz",
                     "images_02.tar.gz",
                     "images_03.tar.gz",
                     "images_04.tar.gz",
                     "images_05.tar.gz",
                     "images_06.tar.gz",
                     "images_07.tar.gz",
                     "images_08.tar.gz",
                     "images_09.tar.gz",
                     "images_10.tar.gz",
                     "images_11.tar.gz",
                     "images_12.tar.gz",
                     ]
        for tar_split in tarsplits_list:
            with tarfile.open(os.path.join(self.source_dir, tar_split)) as tar:
                tar.extractall()


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
                    if not any(label):
                        print(findings)
                    label_list.append(label)
        label_tensors = torch.LongTensor(label_list)
        os.makedirs(self.index_cache_path, exist_ok=True)
        torch.save({'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list},
                   osp.join(self.index_cache_path, self.cache_file))
        return

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


class NIHChest(AbstractDomainInterface):
    name = "NIHCC"
    def __init__(self, root_path="./workspace/datasets/NIHCC", leave_out_classes=(), keep_in_classes=None, binary=False, downsample=None, download=False, extract=True):
        """
        :param leave_out_classes: if a sample has ANY class from this list as positive, then it is removed from indices.
        :param keep_in_classes: when specified, if a sample has None of the class from this list as positive, then it
         is removed from indices..
        """
        super(NIHChest, self).__init__()
        self.leave_out_classes = leave_out_classes
        self.keep_in_classes = keep_in_classes
        self.binary = binary
        self.downsample = downsample
        cache_path = root_path
        source_path = root_path
        if downsample is not None:
            transform = transforms.Compose([transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.Resize((256, 256)),
                                                  transforms.RandomCrop((224, 224)),
                                                  transforms.ToTensor()])
        self.ds_train = NIHChestBase(cache_path, source_path, "train", transforms=transform, binary=self.binary, download=download, extract=extract)
        self.ds_valid = NIHChestBase(cache_path, source_path, "val", transforms=transform, binary=self.binary, download=download, extract=extract)
        self.ds_test = NIHChestBase(cache_path, source_path, "test", transforms=transform, binary=self.binary, download=download, extract=extract)
        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True)
            self.D1_test_ind = self.get_filtered_inds(self.ds_test, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_test)
            self.image_size = (224, 224)

    def get_filtered_inds(self, basedata: NIHChestBase, shuffle=False):
        if not (self.leave_out_classes == () and self.keep_in_classes is None):
            leave_out_mask_label = torch.zeros(N_CLASS).int()
            for cla in self.leave_out_classes:
                ii = CLASSES.index(cla)
                leave_out_mask_label[ii] = 1
            if self.keep_in_classes is None:
                keep_in_mask_label = torch.ones(N_CLASS).int()
            else:
                keep_in_mask_label = torch.zeros(N_CLASS).int()
                for cla in self.keep_in_classes:
                    ii = CLASSES.index(cla)
                    keep_in_mask_label[ii] = 1
            keep_inds = []
            for seq_ind, base_ind in enumerate(basedata.split_inds):
                label = basedata.label_tensors[base_ind]
                if torch.sum(label * leave_out_mask_label) == 0 and torch.sum(label * keep_in_mask_label) > 0:
                    keep_inds.append(seq_ind)
                else:
                    pass
            output_inds = torch.Tensor(keep_inds).int()
        else:
            output_inds = torch.arange(0, len(basedata)).int()
        if shuffle:
            output_inds = output_inds[torch.randperm(len(output_inds))]
            if len(output_inds) > MAX_LENGTH:
                output_inds = output_inds[:MAX_LENGTH]
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
        return transforms.Compose([ExpandRGBChannels(),
                                    transforms.ToPILImage(),
                                   #transforms.Grayscale(),
                                   transforms.Resize((target, target)),
                                   transforms.ToTensor()
                                   ])

class NIHChestBinary(NIHChest):
    def __init__(self):
        super(NIHChestBinary, self).__init__(binary=True)
        return

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
    for batch, batch_ind in zip(loader, range(10)):
        print(batch_ind)
        x, y = batch
        plt.imshow(x.numpy().reshape(dataset.image_size))

    Noeffusion = NIHChest(leave_out_classes=['Effusion'])
    d1_train = Noeffusion.get_D1_train()
    print(len(d1_train))
    loader = data.DataLoader(d1_train, batch_size=1, shuffle=True)
    for batch, batch_ind in zip(loader, range(100)):
        x, y = batch
        #print(y)
        pr_str = ""
        for i, cla in enumerate(CLASSES):
            if y[0][i] == 1:
                pr_str += cla + "|"
        print(pr_str)

    Keepeffusion = NIHChest(keep_in_classes=['Effusion'])
    d1_train = Keepeffusion.get_D1_train()
    print(len(d1_train))
    loader = data.DataLoader(d1_train, batch_size=1, shuffle=True)
    for batch, batch_ind in zip(loader, range(100)):
        x, y = batch
        # print(y)
        pr_str = ""
        for i, cla in enumerate(CLASSES):
            if y[0][i] == 1:
                pr_str += cla + "|"
        print(pr_str)

    NoFibrosisKeepEdemaEffusion = NIHChest(leave_out_classes=['Fibrosis'], keep_in_classes=['Edema', 'Effusion'])
    d1_train = NoFibrosisKeepEdemaEffusion.get_D1_train()
    print(len(d1_train))
    loader = data.DataLoader(d1_train, batch_size=1, shuffle=True)
    for batch, batch_ind in zip(loader, range(100)):
        x, y = batch
        # print(y)
        pr_str = ""
        for i, cla in enumerate(CLASSES):
            if y[0][i] == 1:
                pr_str += cla + "|"
        print(pr_str)
    pass
