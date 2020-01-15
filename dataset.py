import sys
import bisect
from pathlib import Path

import numpy as np
import imageio
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as tfms

from utils import logger
from segmentaion import get_mask, segment

CXR_BASE = Path("/home/jingyi/cxr-jingyi/data").resolve()
STANFORD_CXR_BASE = CXR_BASE.joinpath("stanford/v1").resolve()

def train_val_split(file_path, n_labeled, split=False):

    df = pd.read_csv(str(file_path)).fillna(0)
    labels = np.array(df['target'])
    num_classes = len(np.unique(labels))
    
    if split==True: 
        n_labeled_per_class = int(n_labeled/num_classes)
        train_labeled_idxs = []
        train_unlabeled_idxs = []

        for i in range(num_classes):
            idxs = np.where(labels == i)[0]
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)
        
        train_labeled_df = df.iloc[train_labeled_idxs]
        train_unlabeled_df = df.iloc[train_unlabeled_idxs]

        train_labeled_df.to_csv('data/train_labeled_{}.csv'.format(len(train_labeled_idxs)), index=False)
        train_unlabeled_df.to_csv('data/train_unlabeled_{}.csv'.format(len(train_unlabeled_idxs)), index=False)
        
        print('Write to csv!')
        
    else:
        print('Data unchanged - labeled= {}'.format(n_labeled))

    return num_classes



def _load_manifest(file_path):
    
    df = pd.read_csv(str(file_path)).fillna(0)
    df_tmp = df[['Path','target']]
    entries = df_tmp

    return entries

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512,512), interpolation=3),
    #tfms.Resize(300, Image.LANCZOS),
    tfms.RandomRotation((-10, 10)),
    tfms.RandomCrop((256, 256)),
    tfms.RandomHorizontalFlip(),
    tfms.RandomVerticalFlip(),
    tfms.ToTensor()
])

cxr_test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((512,512), interpolation=3),
    #tfms.Resize(256, Image.LANCZOS),
    tfms.CenterCrop(256),
    tfms.ToTensor()
])

 
def get_image(img_path, transforms):
    image = imageio.imread(img_path[0])
    # transformation
    image_tensor = transforms(image)
    return image_tensor


class CxrDataset(Dataset):

    transforms = cxr_train_transforms

    def __init__(self, base_path, manifest_path, transforms=transforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = _load_manifest(manifest_path)
        self.base_path = base_path
        self.transforms = transforms
       
    def __getitem__(self, index):

        def get_entries(index):
            df = self.entries.loc[index]
            paths = [self.base_path.joinpath(x).resolve() for x in df[0].split(',')]
            label = df[1:].tolist()
            #label = [df[1]]
            return paths, label

        img_path, label = get_entries(index)
        image_tensor = get_image(img_path, self.transforms)
        target_tensor = torch.tensor(label, dtype=torch.long)
       
        return image_tensor, target_tensor

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def train():
        CxrDataset.transforms = cxr_train_transforms

    @staticmethod
    def eval():
        CxrDataset.transforms = cxr_test_transforms


class CXR_unlabeled(Dataset):

    transforms = cxr_train_transforms

    def __init__(self, base_path, manifest_path, transforms=transforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = _load_manifest(manifest_path)
        self.base_path = base_path
        self.transforms = transforms

    def __getitem__(self, index):

        def get_entries(index):
            df = self.entries.loc[index]
            paths = [self.base_path.joinpath(x).resolve() for x in df[0].split(',')]
            label = df[1:].tolist()
            #label = [df[1]]
            return paths, label

        img_path, label = get_entries(index)
        image_tensor = get_image(img_path, TransformTwice(self.transforms))
        target_tensor = torch.tensor(label, dtype=torch.long)
    
        return image_tensor, target_tensor

    def __len__(self):
        return len(self.entries)
