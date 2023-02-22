import pandas as pd
import os
import torch

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

from PIL import Image

class FOODDataset(VisionDataset):
    def __init__(self, root, train=True):
        self.transform = None
        self.target_transform = None
        
        self.loader = default_loader
        
        split = 'train' if train is True else 'val'
        
        dataframe = pd.read_csv(os.path.join(root, f'annot/{split}_info.csv'),
                                names= ['image_name','label'])
        dataframe['path'] = dataframe['image_name'].map(lambda x: os.path.join(f'{root}/{split}_set/', x))
        
        self.samples = [(row['path'], row['label']) for _, row in dataframe.iterrows()]
        
        with open(os.path.join(root, 'annot', 'class_list.txt')) as f:
            self.num_classes = len(f.readlines())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target