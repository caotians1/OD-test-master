import numpy as np
from os.path import join
import pandas as pd
import pickle
from PIL import Image

from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def split_dataset(csvpath: str, output: str, train=0.6, val=0.2, seed=666) -> None:
    """
    Split the data contained in csvpath in train/val/test, and write the results in output.
    """
    df = pd.read_csv(csvpath)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    patients_ids = df.PatientID.unique()

    train_val_split_idx = int(train * len(patients_ids))
    val_test_split_idx = int((train + val) * len(patients_ids))

    train_ids = patients_ids[:train_val_split_idx]
    val_ids = patients_ids[train_val_split_idx:val_test_split_idx]
    test_ids = patients_ids[val_test_split_idx:]

    with open(output, 'wb') as f:
        pickle.dump((train_ids, val_ids, test_ids), f)


class PCXRayDataset(Dataset):
    def __init__(self, datadir, csvpath, splitpath=None, transform=None, views=["PA","L"],
                 dataset='train', pretrained=False, min_patients_per_label=50, 
                 counter_examples=False, duplicate=False):
        """
        Data reader. Only selects labels that at least min_patients_per_label patients have.
        """
        super(PCXRayDataset, self).__init__()

        assert dataset in ['train', 'val', 'test']

        self.datadir = datadir
        self.transform = transform
        self.pretrained = pretrained
        self.threshold = min_patients_per_label
        self.views = views
        self.counter_examples = counter_examples
        self.duplicate = duplicate

        self.df = pd.read_csv(csvpath)
        self.total_samples = len(self.df.PatientID.unique())

        self._build_labels()
        self.mb = MultiLabelBinarizer(classes=self.labels)
        self.mb.fit(self.labels)

        # Split into train or validation
        if splitpath is not None:
            with open(splitpath, 'rb') as f:
                train_ids, val_ids, test_ids = pickle.load(f)
            if dataset == 'train':
                self.df = self.df[self.df.PatientID.isin(train_ids)]
            elif dataset == 'val':
                self.df = self.df[self.df.PatientID.isin(val_ids)]
            else:
                self.df = self.df[self.df.PatientID.isin(test_ids)]

            self.df = self.df.reset_index()
            
        self.nb_classes = len(self.labels)
        
        if self.duplicate:
            self.to_duplicate = np.random.choice(range(self.total_samples), self.total_samples//2)
            self.nb_classes = self.nb_classes*2
        
    def __len__(self):
        return self.total_samples

    
    def get_labels(self, idx):
        subset = self.df[self.df.PatientID == self.df.PatientID[idx * 2]]
        labels = eval(subset.Clean_Labels.tolist()[0])
        if set(labels).difference(self.labels):
            labels.append('other')
        labels = [l for l in labels if l in self.labels]
        return labels
    
    def __getitem__(self, idx):

        subset = self.df[self.df.PatientID == self.df.PatientID[idx * 2]]
        labels = self.get_labels(idx)
        encoded_labels = self.mb.transform([labels]).squeeze()
        
        sample = {}
        if "PA" in self.views:
            pa_path = subset[subset.Projection == 'PA'][['ImageID', 'ImageDir']]
            pa_path = join(self.datadir, pa_path['ImageID'].tolist()[0])
            #pa_path = join(self.datadir,'216840111366964012989926673512011108125227151_00-185-152.png')
            pa_img = np.array(Image.open(pa_path))[..., np.newaxis]
            if self.pretrained:
                pa_img = np.repeat(pa_img, 3, axis=-1)
            sample["PA"] = pa_img

        if "L" in self.views:
            l_path = subset[subset.Projection == 'L'][['ImageID', 'ImageDir']]
            l_path = join(self.datadir, l_path['ImageID'].tolist()[0])
            # l_path = './data/processed/0/46523715740384360192496023767246369337_veyewt.png'
            l_img = np.array(Image.open(l_path))[..., np.newaxis]
            if self.pretrained:
                l_img = np.repeat(l_img, 3, axis=-1)
            sample["L"] = l_img

        if self.transform is not None:
            sample = self.transform(sample)

        sample['labels'] = labels
        sample['encoded_labels'] = torch.from_numpy(encoded_labels.astype(np.float32))
        sample['sample_weight'] = torch.max(sample['encoded_labels'] * self.labels_weights)
                        
        
        if self.counter_examples:
            
            if self.duplicate:
                new_encoded_labels = torch.zeros(self.nb_classes).long()
                
                # put the labels at the lower or upper copy
                if idx in self.to_duplicate:
                    new_encoded_labels[:self.nb_classes//2] = sample['encoded_labels']
                else:
                    new_encoded_labels[self.nb_classes//2:] = sample['encoded_labels']
                
                sample['encoded_labels'] = new_encoded_labels
               
            # pick a query label
            topresent = np.random.choice(range(len(sample['encoded_labels'])), p=self.labels_weights_dup)
            sample['cond'] = torch.LongTensor([topresent]).squeeze()
            
            # is the query in this sample?
            sample['cond_target'] = sample['encoded_labels'][topresent].long().squeeze()
            
            sample['cond_weight'] = self.labels_weights[sample['cond']//2]

        return sample["PA"], sample['cond_target'], sample['cond'], sample['cond_weight']

    def _build_labels(self):
        labels_dict = {}
        for labels in self.df.Clean_Labels:
            for label in eval(labels):
                label = label.strip()
                if label not in labels_dict:
                    labels_dict[label] = 0
                labels_dict[label] += 1

        labels = []
        labels_count = []
        other_counts = []
        for k, v in labels_dict.items():
            if v > self.threshold * 2:
                labels.append(k)
                labels_count.append(v)
            else:
                other_counts.append(v)
                
        labels.append('other')
        labels_count.append(sum(other_counts))
        
        self.labels = labels
        self.labels_count = labels_count
        self.labels_weights = torch.from_numpy(np.array([(len(self) / label)
                                                         for label in labels_count], dtype=np.float32))
        self.labels_weights = self.labels_weights/self.labels_weights.max()
        self.labels_weights = self.labels_weights**2
        self.labels_weights_dup = torch.cat([self.labels_weights]*2).numpy()
        self.labels_weights_dup /= self.labels_weights_dup.sum()
        #self.labels_weights = torch.clamp(self.labels_weights * 0.1, 1., 5.)
        self.nb_labels = len(self.labels)


class Normalize(object):
    """
    Changes images values to be between -1 and 1.
    """
    def __call__(self, sample):
        
        if 'PA' in sample:
            pa_img = sample['PA']
            pa_img = 2 * (pa_img / 65536) - 1.
            pa_img = pa_img.astype(np.float32)
            sample['PA'] = pa_img
            
        if 'L' in sample:    
            l_img = sample['L']
            l_img = 2 * (l_img / 65536) - 1.
            l_img = l_img.astype(np.float32)
            sample['L'] = l_img

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        to_tensor = transforms.ToTensor()
        if 'PA' in sample:
            sample['PA'] = to_tensor(sample['PA'])
        if 'L' in sample: 
            sample['L'] = to_tensor(sample['L'])

        return sample


class ToPILImage(object):
    """
    Convert ndarrays in sample to PIL images.
    """
    def __call__(self, sample):
        to_pil = transforms.ToPILImage()
        sample['PA'] = to_pil(sample['PA'])
        sample['L'] = to_pil(sample['L'])

        return sample


class GaussianNoise(object):
    """
    Adds Gaussian noise to the PA and L (mean 0, std 0.05)
    """
    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        pa_img += torch.randn_like(pa_img) * 0.05
        l_img += torch.randn_like(l_img) * 0.05

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


class RandomRotation(object):
    """
    Adds a random rotation to the PA and L (between -5 and +5).
    """
    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        rot_amount = np.random.rand() * 5.
        rot = transforms.RandomRotation(rot_amount)
        pa_img = rot(pa_img)
        l_img = rot(l_img)

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample
		