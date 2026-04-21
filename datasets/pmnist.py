import torch
import numpy as np
from torchvision import datasets, transforms
from .dataset import Dataset


class pMNIST(Dataset):
    """Class to generate the MNIST dataset with some properties"""

    def __init__(self, **kwargs):
        seq_length = self.image_size * self.channels
        train_size = 50000
        val_size = 10000
        test_size = 10000
        if 'train_size' in kwargs.keys():
            if 'val_size' in kwargs.keys():
                assert kwargs['train_size'] + kwargs['val_size'] == 60000, (
                    "validation and train sets should contain in a whole 60000 examples")
            else: kwargs['val_size'] = 60000 - kwargs['train_size']
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
        return 28 * 28

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

        torch.manual_seed(5544)
        np.random.seed(5544)
        permute = np.random.RandomState(92916)
        permutation = torch.LongTensor(permute.permutation(self.seq_length))

        # this transform allows to download the mnist images in the flattened randomly-permuted shape
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 1)[permutation])
        ])
        
        print("-" * 43 + f" Loading {type(self).__name__} " + "-" * 43)

        train_ds, val_ds = torch.utils.data.random_split(
            datasets.MNIST("data/mnist_data", train=True, download=True, transform=transform),
            [self.train_size, self.val_size]
        )
        test_ds = datasets.MNIST("data/mnist_data", train=False, transform=transform)

        print("-" * 43 + f" {type(self).__name__} loaded " + "-" * 43)

        return train_ds, val_ds, test_ds
