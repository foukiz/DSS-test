import torch
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from PIL import ImageDraw

import cv2
cv2.useOptimized()

try:
    from .dataset import Dataset
except ImportError:
    from dataset import Dataset


class Pathfinder(Dataset):
    
    def __init__(
        self,
        data_dir,
        split,
        **kwargs
    ):
        """ Class to generate the Pathfinder dataset with some properties
            This dataset is used for the 'Pathfinder' benchmark in LRA
        """

        self.data_dir = data_dir
        self.split = split

        data = torch.load(f'{data_dir}/pathfinder_{split}.pt')
        super().__init__(data['images'], data['labels'])

    @property
    def input_dimension(self):
        return (self._input_dimension,)

    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def input_flat_dimension(self):
        return self._input_dimension

    @property
    def image_size(self):
        return (self._input_dimension,)

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return 2

    @num_outputs.setter
    def num_outputs(self, value):
        self._output_dimension = value

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
        #TODO
        pass
