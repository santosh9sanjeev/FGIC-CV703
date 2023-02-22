import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

from .aircraft import Aircraft
from .cars import Cars

class AircraftCars(VisionDataset):
    """`Dataset_Description <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    def __init__(self, root, train=True):
        self.transform = None
        self.target_transform = None
        
        self.loader = default_loader
        
        aircraft = Aircraft(root, transform=None, train=train, download=False, test=True)
        cars     = Cars(os.path.join(root, 'stanford_cars'),
                        transform=None, download=False, train=train,
                        test=True, size=10, idx_start_from=len(aircraft.class_to_idx))
        
        self.samples = []
        self.samples.extend(aircraft.samples)
        self.samples.extend([(os.path.join(cars.root, path), target)
                             for path, target in cars.samples])
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
    def __len__(self):
        return len(self.samples)