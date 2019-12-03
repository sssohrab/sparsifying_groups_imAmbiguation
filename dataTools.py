import torch
import numpy as np
import os
from cv2 import imread,resize, COLOR_BGR2RGB
import matplotlib.pyplot as plt
from torch.utils.data import Dataset #, DataLoader
from torchvision import transforms, utils

#######################################################################
class dataRead(Dataset):

    def __init__(self, root, shape, transform=None):

        self.root = root
        self.shape = shape
        self.transform = transform
        self.image_names = os.listdir(root)

    def __len__(self):
        return int(len(self.image_names))

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        if self.shape[2] == 1:
            image = imread(self.root + image_name,0)
        else:
            image = imread(self.root + image_name, COLOR_BGR2RGB)
            image = image[:,:,::-1]
        image = resize(image,self.shape[0:2])
        sample = {}
        sample['name'] = image_name
        if self.transform:
            sample['image'] = self.transform(image)

        return sample
    
#######################################################################
class dataRead_fromName(Dataset):

    def __init__(self, root, shape, names_list, transform=None):

        self.root = root
        self.shape = shape
        self.names_list = names_list
        self.transform = transform

    def __len__(self):
        return int(sum(1 for line in open(self.names_list)))

    def __getitem__(self, idx):
        with open(self.names_list) as f:
            image_name = f.readlines()[idx][:-1]   # For some damn reason, the last character is space.
        
        if self.shape[2] == 1:
            image = imread(os.path.join(self.root, image_name),0)
        else:
            image = imread(self.root + image_name , COLOR_BGR2RGB)
            image = image[:,:,::-1]
        image = resize(image,self.shape[0:2])
        sample = {}
        sample['name'] = image_name
        if self.transform:
            sample['image'] = self.transform(image)

        return sample