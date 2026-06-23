import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from einops.layers.torch import Rearrange

import os
import numpy as np

from PIL import Image

from pathlib import Path


try:
    from .dataset import Dataset
except ImportError:
    from dataset import Dataset




DATA_DIR = 'data/pathfinder_data'
IMGS_DIR = Path(DATA_DIR) / "imgs/0"



class Pathfinder(Dataset):
    
    def __init__(self, **kwargs):
        seq_length = self.image_size * self.channels
        train_size = kwargs.pop('train_size', 160000)
        val_size = kwargs.pop('val_size', 20000)
        test_size = kwargs.pop('test_size', 20000)
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
        return 2

    #@num_outputs.setter
    #def num_outputs(self, value):
    #    self._output_dimension = value

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
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
            Rearrange("1 h w -> (h w) 1")
        ])

        print("-" * 43 + f" {type(self).__name__} loaded " + "-" * 43)

        pathfinder_ds = PathfinderDataset(transforms=transform)

        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            pathfinder_ds,
            [self.tr_size, self.va_size, self.te_size],
            generator=gen,
        )

        print("-" * 43 + f" {type(self).__name__} loaded " + "-" * 43)

        return train_ds, val_ds, test_ds



class PathfinderDataset(torch.utils.data.Dataset):

    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms
        self.data_dir = DATA_DIR
        self.img_dir = IMGS_DIR

        if os.path.exists(self.data_dir + "/data_pairs.npy"):
            self.pairs = np.load(self.data_dir + "/data_pairs.npy", allow_pickle=True)

        else:
            self.pairs = process_raw_data(save=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, target = self.pairs[index]
        with open(self.img_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, int(target)




def process_raw_data(save=True):
    """ process les données brutes pour stocker des paires (image_path, label) dans un fichier np.save

        les images sont au format "sample_X.png"
    """

    samples = []

    metadata = np.load(DATA_DIR + "/metadata/0.npy", allow_pickle=True)
    for line in metadata:
        samples.append(
            (line[1], int(line[3]))
        )

    if save:
        np.save(DATA_DIR + "/data_pairs.npy", samples)

    return samples




if __name__ == "__main__":
    ds = Pathfinder()
    slice = ds.test_ds.__getitem__(0)
    print(slice)