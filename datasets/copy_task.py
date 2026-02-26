import torch
import numpy as np
from .dataset import Dataset
from torch.utils.data import TensorDataset


class CopyMemory(Dataset):
    """Class to generate the sequence copy task dataset with some properties"""
    def __init__(self, train_size, val_size, test_size, seq_length, sent_length, vocabulary, **kwargs):
        self.vocabulary = vocabulary
        self.sent_length = sent_length
        self.naive_baseline = sent_length * np.log(vocabulary - 2) / seq_length
        super().__init__(train_size, val_size, test_size, seq_length, **kwargs)

    @property
    def input_dimension(self):
        return (self.vocabulary,)
    
    @input_dimension.setter
    def input_dimension(self, value):
        self._input_dimension = value

    @property
    def input_flat_dimension(self):
        return self.vocabulary

    @property
    def image_size(self):
        return None

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return self.input_flat_dimension - 1

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
    
    def __str__(self):
        str_ret = super().__str__()
        str_ret += f"\nVocabulary size: {self.vocabulary}\nCopied sentence length: {self.sent_length}"
        return str_ret

    # Generates Synthetic Data
    def Generate_Data_copy_task(self, size, length, K):

        assert length > 2 * K, (
            "copy_memory sequence length issue: the whole sequence should be "
            "strictly longer than twice its relevant part (the tokens to memorize)"
        )

        dim = self.input_flat_dimension
        seq = torch.randint(1, dim-1, size=(size, K))
        L = length - 2 * K
        zeros1 = torch.zeros((size, L), dtype=torch.int32)
        zeros2 = torch.zeros((size, K-1), dtype=torch.int32)
        zeros3 = torch.zeros((size, K+L), dtype=torch.int32)
        marker = (dim - 1) * torch.ones((size, 1), dtype=torch.int32)

        x = torch.cat((seq, zeros1, marker, zeros2), axis=1)
        
        y = torch.cat((zeros3, seq), axis=1)

        return x, y

    def import_dataset(self):
        torch.manual_seed(0)

        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        x_train, y_train = self.Generate_Data_copy_task(
                                    self.train_size, self.seq_length, self.sent_length)
        x_val, y_val = self.Generate_Data_copy_task(
                                    self.val_size, self.seq_length, self.sent_length)
        x_test, y_test = self.Generate_Data_copy_task(
                                    self.test_size, self.seq_length, self.sent_length)
        
        #train_ds = OneHotBatchDataset(self.input_flat_dimension, x_train, y_train)
        #val_ds = OneHotBatchDataset(self.input_flat_dimension, x_val, y_val)
        #test_ds = OneHotBatchDataset(self.input_flat_dimension, x_test, y_test)
        
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)

        print("-" * 60 + f"{type(self).__name__} loaded" + "-" * 60)

        return train_ds, val_ds, test_ds




class OneHotBatchDataset(TensorDataset):

    def __init__(self, num_classes, *tensors: torch.Tensor) -> None:
        super().__init__(*tensors)
        self.num_classes = num_classes

    def __getitem__(self, index):
        batch_x, batch_y = super().__getitem__(index)
        # one hot encode the batched sequence to predict
        batch_x = torch.nn.functional.one_hot(batch_x, num_classes=self.num_classes).to(torch.float32)
        return (batch_x, batch_y)


if __name__ == '__main__':

    CopyMemory(1000,200,200,20,3,4)