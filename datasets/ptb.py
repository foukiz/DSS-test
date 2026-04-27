import kagglehub
import torch
import os
import pickle as pkl
import pandas as pd
import itertools

try:
  from .dataset import Dataset
except ImportError:
  from dataset import Dataset

from torch.utils.data import TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import utils



DATA_DIR = 'data/ptb_data'

TRAIN_SIZE = 42068
VAL_SIZE = 3370
TEST_SIZE = 3761


class PennTreebank(Dataset):
  
    def __init__(
        self,
        seq_length=150,
        data_dir=DATA_DIR,
        **kwargs
    ):
        self.data_dir = data_dir
        self.vocab = None
        train_size = kwargs.pop('train_size', TRAIN_SIZE)
        val_size = kwargs.pop('val_size', VAL_SIZE)
        test_size = kwargs.pop('test_size', TEST_SIZE)
        super().__init__(train_size, val_size, test_size, seq_length=seq_length, **kwargs)

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
        return None

    @property
    def channels(self):
        return 1

    @property
    def num_outputs(self):
        return self._output_dimension

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

        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        if os.path.exists(self.data_dir + '/vocab.pkl'):
            with open(self.data_dir + '/vocab.pkl', "rb") as f: self.vocab = pkl.load(f)
        else:
            self.vocab = make_vocab(data_dir=self.data_dir, save=True)
        self.input_dimension = len(self.vocab)
        self.num_outputs = len(self.vocab)
        self.padding_idx = self.vocab["<pad>"]

        if not os.path.exists(self.data_dir + '/data.pkl'):
            x_train, y_train = process_raw_data(vocab=self.vocab, max_len=self.seq_length, kind='train')
            x_val, y_val = process_raw_data(vocab=self.vocab, max_len=self.seq_length, kind='val')
            x_test, y_test = process_raw_data(vocab=self.vocab, max_len=self.seq_length, kind='test')
            data = {
                'train_ds': (x_train, y_train),
                'val_ds': (x_val, y_val),
                'test_ds': (x_test, y_test)
            }
            with open(self.data_dir + "/data.pkl", "wb") as f: pkl.dump(data, f)
        else:
            with open(self.data_dir + "/data.pkl", "rb") as f: data = pkl.load(f)
            x_train, y_train = data['train_ds']
            x_val, y_val = data['val_ds']
            x_test, y_test = data['test_ds']

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)
        test_ds = TensorDataset(x_test, y_test)

        print("-" * 60 + f"{type(self).__name__} loaded" + "-" * 60)

        return train_ds, val_ds, test_ds







""" PROCESSING DATA """

def make_vocab(append_bos=False, append_eos=False, data_dir=DATA_DIR, save=True):
    """ Return the vocabulary (an instance of Vocab) made out of the ListOps validation file
    """

    with open(DATA_DIR + "/ptb.train.txt", "r", encoding="utf-8") as f: lines = f.readlines()

    print("Building vocab...")

    vocab = utils.build_vocab(
        lines,
        specials=["<pad>", "<unk>"] + (["<bos>"] if append_bos else []) + (["<eos>"] if append_eos else [])
    )
    vocab.set_default_index(vocab["<unk>"])

    if save:
        with open(data_dir + "/vocab.pkl", "wb") as f: pkl.dump(vocab, f)

    return vocab


def process_raw_data(
    vocab,
    max_len=150,
    kind='train',
    data_dir=DATA_DIR
):
    # process raw data into tensors
    print(f"Processing {kind} data...")
    with open(data_dir + f"/ptb.{kind}.txt", "r", encoding="utf-8") as f:
        df = pd.DataFrame({
            "input": (list(line.strip()) for line in f if line.strip())
        })

    df['target'] = df['input'].apply(lambda x: x[1:])

    numericalize = lambda example: vocab(example)[:max_len]
    df['input'] = df['input'].apply(numericalize)
    df['target'] = df['target'].apply(numericalize)

    padding = lambda x: utils.pad_sequence(x, max_len=max_len, pad_val=vocab['<pad>'])
    df['input'] = df['input'].apply(padding)
    df['target'] = df['target'].apply(padding)

    inputs = torch.tensor(df['input'], dtype=torch.int32)
    targets = torch.tensor(df['target'], dtype=torch.int64)

    print("preprocessing done.")

    return inputs, targets




if __name__ == "__main__":
    PennTreebank()
