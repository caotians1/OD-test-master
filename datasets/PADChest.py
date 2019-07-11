import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface, ExpandRGBChannels
import os
import os.path as osp
import csv
import subprocess
from PIL import Image

N_CLASS = 2
MAX_LENGTH = 1000000
def to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def group_normalize(crops):
    return torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])(crop) for crop in crops])

class PADChestBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir,
                 index_file="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", image_dir="images_64",
                 imsize=224, transforms=None, binary=False, to_rgb=False, download=False, extract=True):
        super(PADChestBase,self).__init__()
        self.index_cache_path = index_cache_path
        self.source_dir = source_dir
        self.cache_file = "PADChestIndex.pkl"
        self.index_file = index_file
        self.image_dir = image_dir
        self.imsize = imsize
        self.binary = binary
        self.to_rgb = to_rgb
        if transforms is None:
            self.transforms = transforms.Compose([transforms.Resize((imsize,imsize)),
                                                transforms.ToTensor()])
        else:
            self.transforms = transforms

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
        imp = osp.join(self.source_dir, self.image_dir, img_name)
        with open(imp, 'rb') as f:
            with Image.open(f) as img:
                if self.to_rgb:
                    img = self.transforms(img.convert('RGB'))
                else:
                    img = self.transforms(img.convert('L'))
        return img, label

    def extract(self):
        if os.path.exists(os.path.join(self.source_dir, self.image_dir)):
            return
        import tarfile
        tarsplits_list = ["images_01.tar.gz",
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
                if not row['Projection'] in ['L', 'PA']:
                    continue
                imp = osp.join(self.source_dir, self.image_dir, row['ImageID'])
                if osp.exists(imp):
                    img_list.append(row['ImageID'])
                    label = [0, 1] if 'L' in row['Projection'] else [1, 0]
                    label_list.append(label)
        label_tensors = torch.LongTensor(label_list)
        os.makedirs(self.index_cache_path, exist_ok=True)
        torch.save({'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list},
                   osp.join(self.index_cache_path, self.cache_file))
        return


class PADChest(AbstractDomainInterface):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True):

        self.name = "PADChest"
        super(PADChest, self).__init__()
        self.downsample = downsample
        self.expand_channels=expand_channels
        self.max_l = test_length
        cache_path = root_path
        source_path = root_path
        if downsample is not None:
            print("downsampling to", downsample)
            transform = transforms.Compose([transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor()])
            self.image_size = (downsample, downsample)
        else:
            transform = transforms.Compose([transforms.RandomCrop((64, 64)),
                                            transforms.ToTensor()])
            self.image_size = (64, 64)

        self.ds_all = PADChestBase(cache_path, source_path, transforms=transform,
                                     to_rgb=expand_channels, download=download, extract=extract)
        n_train = int(0.6*len(self.ds_all))     # A generous 6-4 split since we don't use the train split at all...
        n_val = len(self.ds_all) - n_train
        self.ds_train, self.ds_valid = data.random_split(self.ds_all, [n_train, n_val])

        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True, max_l=self.max_l)
            self.D1_test_ind = self.get_filtered_inds(self.ds_valid, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_valid, shuffle=True)


    def get_filtered_inds(self, basedata: PADChestBase, shuffle=False, max_l=None):
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
        target = 64
        if self.downsample is not None:
            target = self.downsample
        if self.expand_channels:
            return transforms.Compose([ExpandRGBChannels(),
                                        transforms.ToPILImage(),
                                       #transforms.Grayscale(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor()
                                       ])
        else:
            return transforms.Compose([
                                       transforms.ToPILImage(),
                                       transforms.Grayscale(),
                                       transforms.Resize((target, target)),
                                       transforms.ToTensor()
                                       ])


if __name__ == "__main__":
    dataset = PADChest()
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
