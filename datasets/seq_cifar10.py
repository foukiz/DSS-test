import torch
import numpy as np
from torchvision import datasets, transforms
from .dataset import Dataset


class sCIFAR10(Dataset):
    """Class to generate the sequential CIFAR10 dataset with some properties"""

    def __init__(self, **kwargs):
        seq_length = self.image_size * self.channels
        train_size = 45000
        val_size = 5000
        test_size = 10000
        if 'train_size' in kwargs.keys():
            if 'val_size' in kwargs.keys():
                assert kwargs['train_size'] + kwargs['val_size'] == 50000, (
                    "validation and train sets should contain in a whole 50000 examples")
            else: kwargs['val_size'] = 50000 - kwargs['train_size']
            train_size = kwargs['train_size']
            val_size = kwargs['val_size']
            kwargs.pop('train_size')
            kwargs.pop('val_size')
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)

    @property
    def input_dimension(self):
        return (1,)

    @property
    def input_flat_dimension(self):
        return 1

    @property
    def image_size(self):
        return 32 * 32

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return 10

    @property
    def test_size(self):
        return self.te_size

    @property
    def train_size(self):
        return self.tr_size

    @property
    def val_size(self):
        return self.va_size

    def get_train_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.test_ds

    def get_val_ds(self):
        return self.val_ds

    def import_dataset(self):

        # this transform allows to download the cifar10 images in the flattened shape
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(-1, 1))
        ])
        
        print("-" * 43 + f" Loading {type(self).__name__} " + "-" * 43)

        train_ds, val_ds = torch.utils.data.random_split(
            datasets.CIFAR10("cifar10_data/cifar10", train=True, download=True, transform=transform),
            [self.train_size, self.val_size]
        )
        test_ds = datasets.CIFAR10("cifar10_data/cifar10", train=False, download=True, transform=transform)

        print("-" * 43 + f" {type(self).__name__} loaded " + "-" * 43)

        return train_ds, val_ds, test_ds
