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

N_CLASS = 5
N_CLASS_SV = 15
CLASSES = ["PA", "AP", "L", "AP_horizontal", "PED" ]
SVCLASSES = ['pleural effusion', 'emphysema', 'pneumonia', 'cardiomegaly', 'pneumothorax', 'mass', 'infiltrates', 'normal',
           'nodule', 'consolidation', 'atelectasis', 'pulmonary edema', 'pulmonary fibrosis', 'hiatal hernia', 'pleural thickening']
MAX_LENGTH = 1000000
def to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

def group_normalize(crops):
    return torch.stack([transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])(crop) for crop in crops])

class PADChestBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir,
                 index_file="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", image_dir="images-64",
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

    def __len__(self):
        return len(self.label_tensors)

    def __getitem__(self, item):

        img_name = self.img_list[item]
        label = self.label_tensors[item]
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
        tarsplits_list = ["images-64.tar",
                            ]
        for tar_split in tarsplits_list:
            with tarfile.open(os.path.join(self.source_dir, tar_split)) as tar:
                tar.extractall(os.path.join(self.source_dir, self.image_dir))


    def generate_index(self):
        """
        Scan index file to create list of images and labels for each image. Also stores index files in index_cache_path
        :return:
        """
        img_list = []
        label_list = []
        disease_label_list = []
        with open(osp.join(self.source_dir, self.index_file), 'r') as fp:
            csvf = csv.DictReader(fp)
            for row in csvf:
                if not row['Projection'] in ['L', 'PA', 'AP', 'AP_horizontal']:
                    continue
                imp = osp.join(self.source_dir, self.image_dir, row['ImageID'])
                if osp.exists(imp):
                    img_list.append(row['ImageID'])
                    if row['Pediatric'] == "PED":
                        label = np.zeros(5, dtype=np.int64)
                        label[4] = 1
                    else:
                        label = np.zeros(5, dtype=np.int64)
                        ind = CLASSES.index(row['Projection'])
                        label[ind] = 1
                    label_list.append(label)
        label_tensors = torch.LongTensor(label_list)
        os.makedirs(self.index_cache_path, exist_ok=True)
        torch.save({'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list},
                   osp.join(self.index_cache_path, self.cache_file))
        return


class PADChest(AbstractDomainInterface):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", keep_class=None, downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True, doubledownsample=None):

        self.name = "PADChest"
        super(PADChest, self).__init__()
        self.keep_in_classes = keep_class
        self.downsample = downsample
        self.expand_channels=expand_channels
        self.max_l = test_length
        cache_path = root_path
        source_path = root_path
        if doubledownsample is not None:
            transform_list = [transforms.Resize(doubledownsample),]
        else:
            transform_list = []
        if downsample is not None:
            print("downsampling to", downsample)
            transform = transforms.Compose(transform_list +[
                                            transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor()])
            self.image_size = (downsample, downsample)
        else:
            transform = transforms.Compose(transform_list +[transforms.Resize((64, 64)),
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
        if not self.keep_in_classes is None:
            #print(basedata.__dict__)
            keep_in_mask_label = torch.zeros(N_CLASS).int()
            for cla in self.keep_in_classes:
                ii = CLASSES.index(cla)
                keep_in_mask_label[ii] = 1
            keep_inds = []
            for seq_ind, base_ind in enumerate(basedata.indices):
                label = basedata.dataset.label_tensors[base_ind].int()
                if torch.sum(label * keep_in_mask_label) > 0:
                    keep_inds.append(seq_ind)
                else:
                    pass
            output_inds = torch.Tensor(keep_inds).int()
        else:
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
        return SubDataset(self.name, self.ds_valid, self.D1_test_ind, label=0)

    def get_D2_valid(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_valid_ind
        return SubDataset(self.name, self.ds_train, target_indices, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_test_ind
        return SubDataset(self.name, self.ds_valid, target_indices, label=1, transform=D1.conformity_transform())

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


class PADChestAP(PADChest):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True, doubledownsample=None):
        super(PADChestAP, self).__init__(root_path, ["AP",], downsample, expand_channels,
                 test_length, download, extract, doubledownsample)


class PADChestL(PADChest):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True, doubledownsample=None):
        super(PADChestL, self).__init__(root_path, ["L",], downsample, expand_channels,
                 test_length, download, extract, doubledownsample)


class PADChestAPHorizontal(PADChest):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True, doubledownsample=None):
        super(PADChestAPHorizontal, self).__init__(root_path, ["AP_horizontal",], downsample, expand_channels,
                 test_length, download, extract, doubledownsample)


class PADChestPED(PADChest):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", downsample=None, expand_channels=False,
                 test_length=None, download=False, extract=True, doubledownsample=None):
        super(PADChestPED, self).__init__(root_path, ["PED",], downsample, expand_channels,
                                                   test_length, download, extract, doubledownsample)


class PADChestSVBase(data.Dataset):
    def __init__(self, index_cache_path, source_dir,
                 index_file="PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", image_dir="images-299",
                 imsize=224, transforms=None, binary=False, to_rgb=False, download=False, extract=True):
        super(PADChestSVBase,self).__init__()
        self.index_cache_path = index_cache_path
        self.source_dir = source_dir
        self.cache_file = "PADChestSVIndex.pkl"
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

    def __len__(self):
        return len(self.label_tensors)

    def __getitem__(self, item):
        img_name = self.img_list[item]
        label = self.label_tensors[item]
        if self.binary:
            label = label[7] * (torch.sum(label) == 1)
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
        if self.image_dir == "images-64":
            tarsplits_list = ["images-64.tar",
                                ]
        else:
            tarsplits_list = ["images-299.tar.gz",
                              ]
        for tar_split in tarsplits_list:
            with tarfile.open(os.path.join(self.source_dir, tar_split)) as tar:
                tar.extractall(os.path.join(self.source_dir, self.image_dir))

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
                if not row['Projection'] == 'L':
                    continue
                labels = row['Labels'].strip("[]").split(',')
                labels = [l.strip("\\ '") for l in labels]
                if not any(l in SVCLASSES for l in labels):
                    continue
                imp = osp.join(self.source_dir, self.image_dir, row['ImageID'])
                if osp.exists(imp):
                    img_list.append(row['ImageID'])
                    numlabel = [1 if cond in labels else 0 for cond in SVCLASSES]
                    label_list.append(numlabel)
        label_tensors = torch.LongTensor(label_list)
        os.makedirs(self.index_cache_path, exist_ok=True)
        torch.save({'img_list': img_list, 'label_tensors': label_tensors, 'label_list': label_list},
                   osp.join(self.index_cache_path, self.cache_file))
        return


class PADChestSV(AbstractDomainInterface):
    dataset_path = "PADChest"
    def __init__(self, root_path="./workspace/datasets/PADChest", leave_out_classes=(), keep_in_classes=None,
                 binary=False, downsample=None, expand_channels=False, test_length=None, download=False,
                 extract=True, doubledownsample=None):
        """
        :param leave_out_classes: if a sample has ANY class from this list as positive, then it is removed from indices.
        :param keep_in_classes: when specified, if a sample has None of the class from this list as positive, then it
         is removed from indices..
        """
        self.name = "PADChest"
        super(PADChestSV, self).__init__()
        self.leave_out_classes = leave_out_classes
        self.keep_in_classes = keep_in_classes
        self.binary = binary
        self.downsample = downsample
        self.expand_channels=expand_channels
        self.max_l = test_length
        cache_path = root_path
        source_path = root_path
        if doubledownsample is not None:
            transform_list = [transforms.Resize(doubledownsample),]
        else:
            transform_list = []
        img_dir = "images-299"
        if downsample is not None:
            print("downsampling to", downsample)
            transform = transforms.Compose(transform_list +[
                                            transforms.Resize((downsample, downsample)),
                                            transforms.ToTensor()])
            self.image_size = (downsample, downsample)
            if downsample == 64:
                img_dir = "images-64"
        else:
            transform = transforms.Compose(transform_list +[transforms.Resize((224, 224)),
                                            transforms.ToTensor()])
            self.image_size = (224, 224)

        self.ds_all = PADChestSVBase(cache_path, source_path, transforms=transform, binary=self.binary,
                                   to_rgb=expand_channels, download=download, extract=extract, image_dir=img_dir)
        n_train = int(0.8 * len(self.ds_all))
        n_val = int(0.1 * len(self.ds_all))
        n_test = len(self.ds_all) - n_train - n_val
        self.ds_train, self.ds_valid, self.ds_test = data.random_split(self.ds_all, [n_train, n_val, n_test])

        if extract:
            self.D1_train_ind = self.get_filtered_inds(self.ds_train, shuffle=True)
            self.D1_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True, max_l=self.max_l)
            self.D1_test_ind = self.get_filtered_inds(self.ds_test, shuffle=True)

            self.D2_valid_ind = self.get_filtered_inds(self.ds_valid, shuffle=True)
            self.D2_test_ind = self.get_filtered_inds(self.ds_test, shuffle=True)


    def get_filtered_inds(self, basedata, shuffle=False, max_l=None):
        if not (self.leave_out_classes == () and self.keep_in_classes is None):
            leave_out_mask_label = torch.zeros(N_CLASS_SV).int()
            for cla in self.leave_out_classes:
                ii = SVCLASSES.index(cla)
                leave_out_mask_label[ii] = 1
            if self.keep_in_classes is None:
                keep_in_mask_label = torch.ones(N_CLASS_SV).int()
            else:
                keep_in_mask_label = torch.zeros(N_CLASS_SV).int()
                for cla in self.keep_in_classes:
                    ii = SVCLASSES.index(cla)
                    keep_in_mask_label[ii] = 1
            keep_inds = []
            for seq_ind, base_ind in enumerate(basedata.indices):
                label = basedata.dataset.label_tensors[base_ind].int()
                if torch.sum(label * leave_out_mask_label) == 0 and torch.sum(label * keep_in_mask_label) > 0:
                    keep_inds.append(seq_ind)
                else:
                    pass
            output_inds = torch.Tensor(keep_inds).long()
        else:
            output_inds = torch.arange(0, len(basedata)).long()
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
        return SubDataset(self.name, self.ds_valid, target_indices, label=1, transform=D1.conformity_transform())

    def get_D2_test(self, D1):
        assert self.is_compatible(D1)
        target_indices = self.D2_test_ind
        return SubDataset(self.name, self.ds_test, target_indices, label=1, transform=D1.conformity_transform())

    def conformity_transform(self):
        target = 224
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


class PADChestBinaryTrainSplit(PADChestSV):
    def __init__(self, *args, **kwargs):
        kwargs.update({'binary': True, 'test_length': 5000,
                          'leave_out_classes':['cardiomegaly', 'pneumothorax', 'nodule', 'mass']})
        super(PADChestBinaryTrainSplit, self).__init__(*args, **kwargs)


class PADChestBinaryValSplit(PADChestSV):
    def __init__(self, *args, **kwargs):
        kwargs.update({'binary': True, 'test_length': 5000,
                       'keep_in_classes': ['cardiomegaly',]})
        super(PADChestBinaryValSplit, self).__init__(*args, **kwargs)


class PADChestBinaryTestSplit(PADChestSV):
    def __init__(self, *args, **kwargs):
        kwargs.update({'binary': True, 'test_length': 5000,
                       'keep_in_classes': ['pneumothorax', 'nodule', 'mass']})
        super(PADChestBinaryTestSplit, self).__init__(*args, **kwargs)

all_labels = ['normal',
 'pulmonary fibrosis',
 'chronic changes',
 'kyphosis',
 'pseudonodule',
 'ground glass pattern',
 'unchanged',
 'alveolar pattern',
 'interstitial pattern',
 'laminar atelectasis',
 'pleural effusion',
 'apical pleural thickening',
 'suture material',
 'sternotomy',
 'endotracheal tube',
 'infiltrates',
 'heart insufficiency',
 'hemidiaphragm elevation',
 'superior mediastinal enlargement',
 'aortic elongation',
 'scoliosis',
 'sclerotic bone lesion',
 'supra aortic elongation',
 'vertebral degenerative changes',
 'goiter',
 'COPD signs',
 'air trapping',
 'descendent aortic elongation',
 'aortic atheromatosis',
 'metal',
 'hypoexpansion basal',
 'abnormal foreign body',
 'central venous catheter via subclavian vein',
 'central venous catheter',
 'vascular hilar enlargement',
 'pacemaker',
 'atelectasis',
 'vertebral anterior compression',
 'hiatal hernia',
 'pneumonia',
 'diaphragmatic eventration',
 'consolidation',
 'calcified densities',
 'cardiomegaly',
 'fibrotic band',
 'tuberculosis sequelae',
 'volume loss',
 'bronchiectasis',
 'single chamber device',
 'emphysema',
 'vertebral compression',
 'bronchovascular markings',
 'bullas',
 'hilar congestion',
 'exclude',
 'axial hyperostosis',
 'aortic button enlargement',
 'calcified granuloma',
 'clavicle fracture',
 'pulmonary mass',
 'dual chamber device',
 'increased density',
 'surgery neck',
 'osteosynthesis material',
 'costochondral junction hypertrophy',
 'segmental atelectasis',
 'costophrenic angle blunting',
 'calcified pleural thickening',
 'hyperinflated lung',
 'callus rib fracture',
 'pleural thickening',
 'mediastinal mass',
 'nipple shadow',
 'surgery heart',
 'pulmonary artery hypertension',
 'central vascular redistribution',
 'tuberculosis',
 'nodule',
 'cavitation',
 'granuloma',
 'osteopenia',
 'lobar atelectasis',
 'surgery breast',
 'NSG tube',
 'hilar enlargement',
 'gynecomastia',
 'atypical pneumonia',
 'cervical rib',
 'mediastinal enlargement',
 'major fissure thickening',
 'surgery',
 'azygos lobe',
 'adenopathy',
 'miliary opacities',
 'suboptimal study',
 'dai',
 'mediastinic lipomatosis',
 'surgery lung',
 'mammary prosthesis',
 'humeral fracture',
 'calcified adenopathy',
 'reservoir central venous catheter',
 'vascular redistribution',
 'hypoexpansion',
 'heart valve calcified',
 'pleural mass',
 'loculated pleural effusion',
 'pectum carinatum',
 'subacromial space narrowing',
 'central venous catheter via jugular vein',
 'vertebral fracture',
 'osteoporosis',
 'bone metastasis',
 'lung metastasis',
 'cyst',
 'humeral prosthesis',
 'artificial heart valve',
 'mastectomy',
 'pericardial effusion',
 'lytic bone lesion',
 'subcutaneous emphysema',
 'pulmonary edema',
 'flattened diaphragm',
 'asbestosis signs',
 'multiple nodules',
 'prosthesis',
 'pulmonary hypertension',
 'soft tissue mass',
 'tracheostomy tube',
 'endoprosthesis',
 'post radiotherapy changes',
 'air bronchogram',
 'pectum excavatum',
 'calcified mediastinal adenopathy',
 'central venous catheter via umbilical vein',
 'thoracic cage deformation',
 'obesity',
 'tracheal shift',
 'external foreign body',
 'atelectasis basal',
 'aortic endoprosthesis',
 'rib fracture',
 'calcified fibroadenoma',
 'pneumothorax',
 'reticulonodular interstitial pattern',
 'reticular interstitial pattern',
 'chest drain tube',
 'minor fissure thickening',
 'fissure thickening',
 'hydropneumothorax',
 'breast mass',
 'blastic bone lesion',
 'respiratory distress',
 'azygoesophageal recess shift',
 'ascendent aortic elongation',
 'lung vascular paucity',
 'kerley lines',
 'electrical device',
 'artificial mitral heart valve',
 'artificial aortic heart valve',
 'total atelectasis',
 'non axial articular degenerative changes',
 'pleural plaques',
 'calcified pleural plaques',
 'lymphangitis carcinomatosa',
 'lepidic adenocarcinoma',
 'mediastinal shift',
 'ventriculoperitoneal drain tube',
 'esophagic dilatation',
 'dextrocardia',
 'end on vessel',
 'right sided aortic arch',
 'Chilaiditi sign',
 'aortic aneurysm',
 'loculated fissural effusion',
 'fracture',
 'air fluid level',
 'round atelectasis',
 'mass',
 'double J stent',
 'pneumoperitoneo',
 'abscess',
 'pulmonary artery enlargement',
 'bone cement',
 'pneumomediastinum',
 'catheter',
 'surgery humeral',
 'empyema',
 'nephrostomy tube',
 'sternoclavicular junction hypertrophy',
 'pulmonary venous hypertension',
 'gastrostomy tube',
 'lipomatosis']

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
