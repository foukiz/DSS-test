import pickle as pkl
import numpy as np
import pandas as pd
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Subset
#from torchtext.vocab import build_vocab_from_iterator

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import utils

try:
    from .dataset import Dataset
except ImportError:
    from dataset import Dataset


DATA_DIR = 'data/imdb_data'



class MappedTensorDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor, transform):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx) 
        if self.transform is None:
            return data
        else:
            return self.transform(data)

class IMDB(Dataset):
    """ Class to generate the IMDB dataset with some properties
        This dataset is used for the 'Text' benchmark in LRA
    """

    def __init__(
        self,
        max_len=2048,
        data_dir=DATA_DIR,
        **kwargs
    ):
        self.data_dir = data_dir
        self.vocab = None
        train_size = kwargs.pop('train_size', 25000)
        val_size = kwargs.pop('val_size', 12500)
        test_size = kwargs.pop('test_size', 12500)
        super().__init__(train_size, val_size, test_size, seq_length=max_len, **kwargs)

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

        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        if not os.path.exists(self.data_dir + '/data.pkl'):
            inputs, targets, vocab = process_raw_csv_data(max_len=self.seq_length)
            self.vocab = vocab
            to_pickle(inputs, targets, vocab)
        with open(self.data_dir + '/data.pkl', 'rb') as f:
            dic = pkl.load(f)
            self.vocab = dic['vocab']

        self.input_dimension = len(self.vocab)
        self.padding_idx = self.vocab["<pad>"]

        train_ds = dic["train_ds"]
        val_ds = dic["val_ds"]
        test_ds = dic["test_ds"]

        print("-" * 60 + f"{type(self).__name__} loaded" + "-" * 60)

        return train_ds, val_ds, test_ds







def process_raw_csv_data(
    max_len=2048,
    min_freq=15,
    append_bos=False,
    append_eos=True,
    data_dir=DATA_DIR
):
    df = pd.read_csv(data_dir + '/imdb.csv')
    max_len = max_len - int(append_bos) - int(append_eos)
    tokenize = lambda example: list(example)[:max_len]
    df['review'] = df['review'].apply(tokenize)
    df['sentiment'] = df['sentiment'].map({"negative":0., "positive":1.})
    vocab = utils.build_vocab(
        df['review'],
        min_freq=min_freq,
        specials=["<pad>", "<unk>"] + (["<bos>"] if append_bos else []) + (["<eos>"] if append_eos else [])
    )
    vocab.set_default_index(vocab["<unk>"])

    numericalize = lambda example: vocab(
        (["<bos>"] if append_bos else []) + example + (["<eos>"] if append_eos else []))
    df['review'] = df['review'].apply(numericalize)
    df['review'] = df['review'].apply(
        lambda x: utils.pad_sequence(x, max_len=(max_len+int(append_bos)+int(append_eos)), pad_val=vocab['<pad>']))

    inputs = torch.tensor(df['review'], dtype=torch.int32)
    targets = torch.tensor(df['sentiment'], dtype=torch.int64)

    print("preprocessing done.")

    return inputs, targets, vocab



def to_pickle(inputs, targets, vocab):
    ds = TensorDataset(inputs, targets)
    train_ds = Subset(ds, range(0, 25000))
    val_ds = Subset(ds, range(25000, 37500))
    test_ds = Subset(ds, range(37500, 50000))

    pkl_dic = {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'vocab': vocab
    }

    with open(DATA_DIR + "/data.pkl", "wb") as f:
        pkl.dump(pkl_dic, f)




def main():
    inputs, targets, vocab = process_raw_csv_data()
    to_pickle(inputs, targets, vocab)
    ds = IMDB()
    print()





if __name__ == "__main__":
    main()