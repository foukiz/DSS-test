import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

try:
    from .dataset import Dataset
except ImportError:
    from dataset import Dataset

import pickle as pkl
import os



DATA_DIR = 'data/cifar10_data'


class sCIFAR10(Dataset):
    """ Class to generate the sequential CIFAR10 dataset with some properties
        This dataset is used for the 'Image' benchmark in LRA
    """


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
            #train_size = kwargs['train_size']
            #val_size = kwargs['val_size']
            train_size = kwargs.pop('train_size')
            val_size = kwargs.pop('val_size')
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)

    @property
    def input_dimension(self):
        return (1,)

    @property
    def input_flat_dimension(self):
        return self.input_dimension[0]

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

        print("-" * 43 + f" Loading {type(self).__name__} " + "-" * 43)

        if not os.path.exists(DATA_DIR + "/data.pkl"):
            process_data(train_size=self.train_size, save=True)
        with open(DATA_DIR + "/data.pkl", "rb") as f:
            data = pkl.load(f)
            train_ds = TensorDataset(*data["train_ds"])
            val_ds = TensorDataset(*data["val_ds"])
            test_ds = TensorDataset(*data["test_ds"])

        print("-" * 43 + f" {type(self).__name__} loaded " + "-" * 43)

        return train_ds, val_ds, test_ds




def process_data(train_size=45000, save=True):
    # this transform allows to download the cifar10 images in the flattened shape

    train_val_samples = datasets.CIFAR10("data/cifar10_data", train=True, download=True)
    test_samples = datasets.CIFAR10("data/cifar10_data", train=False, download=True)

    gen = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(train_val_samples), generator=gen)
    train_samples = [train_val_samples[i] for i in permutation[:train_size]]
    val_samples = [train_val_samples[i] for i in permutation[train_size:]]
    test_samples = [test_samples[i] for i in range(len(test_samples))]

    print("converting train images to tensors...")
    x_train, y_train = cifar2tensor(train_samples)
    print("converting validation images to tensors...")
    x_val, y_val = cifar2tensor(val_samples)
    print("converting test images to tensors...")
    x_test, y_test = cifar2tensor(test_samples)
    print("done.")

    data = {
        "train_ds": (x_train, y_train),
        "val_ds": (x_val, y_val),
        "test_ds": (x_test, y_test)
    }
    if save:
        with open(DATA_DIR + "/data.pkl", "wb") as f: pkl.dump(data, f)

    return data


def cifar2tensor(samples):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=122.6/255.0, std=61.0/255.0),
        transforms.Lambda(lambda x: x.view(1, 1024).t())
    ])

    N = len(samples)
    x = torch.empty((N, 1024, 1), dtype=torch.float32)
    y = torch.empty((N,), dtype=torch.long)
    for i, (data, target) in enumerate(samples):
        x[i] = transform(data)
        y[i] = int(target)

    return x, y






if __name__ == "__main__":

    ds = sCIFAR10()
    ds