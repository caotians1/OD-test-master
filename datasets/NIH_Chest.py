import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from datasets import SubDataset, AbstractDomainInterface
import os.path as osp
import csv
from PIL import Image

CLASSES = ['Effusion', 'Emphysema', 'Pneumonia', 'Cardiomegaly', 'Pneumothorax', 'Mass', 'Infiltration', 'No_Finding',
           'Nodule', 'Consolidation', 'Atelectasis', 'Edema', 'Fibrosis', 'Hernia', 'Pleural_Thickening']
N_CLASS = len(CLASSES)

class NIH_ChestBase(data.Dataset):
    def __init__(self, cached_path, source_dir, index_file="Data_Entry_2017.csv", image_dir="images", imsize=256):
        super(NIH_ChestBase,self).__init__()
        self.cached_path = cached_path
        self.source_dir = source_dir
        self.cache_file = "NIH_Proc.pkl"
        self.index_file = index_file
        self.image_dir = image_dir
        self.imsize = imsize
        self.transforms = transforms.Compose([transforms.Grayscale,
                                              transforms.Resize((self.imsize, self.imsize)),
                                              transforms.ToTensor()])
        if not osp.exists(osp.join(self.cached_path, self.cache_file)):
            self.generate_proc()

    def generate_proc(self):
        img_list = []
        label_list = []

        with open(osp.join(self.source_dir, self.index_file), 'r+') as fp:
            csvf = csv.DictReader(fp)
            for row in csvf:
                imp = osp.join(self.source_dir, self.image_dir, row['Image_Index'])
                if osp.exists(imp):
                    img_list.append(imp)
                    findings = row['Finding Labels'].split('|')
                    label = [1 if cond in findings else 0 for cond in CLASSES]
                    label_list.append(label)
        img_tensors = []
        for imp in img_list:
            with open(imp, 'rb+') as fp:
                with Image.open(fp) as img:
                     img_tensors.append(self.transforms(img))
        img_tensors = torch.stack(img_tensors)
        label_tensors = torch.IntTensor(label_list)
        torch.save({'img_tensors':img_tensors, 'label_tensors':label_tensors}, osp.join(self.cached_path, self.cache_file))
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