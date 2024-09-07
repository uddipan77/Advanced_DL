# data/dataset.py

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RotatedMNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        angle = self.angles[index % len(self.angles)]
        rotated_img = transforms.functional.rotate(img, angle)
        angle_label = self.angles.index(angle)
        return rotated_img, angle_label
