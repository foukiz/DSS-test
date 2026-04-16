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



DATA_DIR = 'data/listops_data'



class ListOps(Dataset):
    """Class to generate the ListOps dataset with some properties"""
    def __init__(
        self,
        train_size=96000,
        val_size=2000,
        test_size=2000,
        min_len=500,
        max_len=2000,
        max_depth=10,
        max_args=10,
        generate=False,
        preprocessed=True,
        data_dir=DATA_DIR,
        **kwargs
    ):
        self.max_len = max_len
        self.min_len = min_len
        self.max_depth = max_depth
        self.max_args = max_args
        self.data_dir = data_dir
        self.generate = generate
        self.preprocessed = preprocessed
        self.vocab = None
       
        # reset after tests
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
    
    def __str__(self):
        ret_str = super().__str__()
        ret_str += f"\nMin length: {self.min_len}\nMax length: {self.max_len}\nMax depth: {self.max_depth}\nMax args: {self.max_args}"
        return ret_str

    def get_train_ds(self):
        return self.train_ds

    def get_test_ds(self):
        return self.test_ds

    def get_val_ds(self):
        return self.val_ds

    def import_dataset(self):
        torch.manual_seed(0)
        
        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        if self.generate:
            self.generate_data(write=True)

        if os.path.exists(self.data_dir + '/vocab.pkl'):
            with open(self.data_dir + '/vocab.pkl', "rb") as f: self.vocab = pkl.load(f)
        else:
            self.vocab = make_vocab(data_dir=self.data_dir, save=True)
        self.input_dimension = len(self.vocab)
        self.padding_idx = self.vocab["<pad>"]

        if not os.path.exists(self.data_dir + '/data.pkl'):
            x_train, y_train = process_data(vocab=self.vocab, max_len=self.max_len, kind='train')
            x_val, y_val = process_data(vocab=self.vocab, max_len=self.max_len, kind='val')
            x_test, y_test = process_data(vocab=self.vocab, max_len=self.max_len, kind='test')
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


def listops_tokenizer(s):
    return s.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None}).split()

def make_vocab(append_bos=False, append_eos=True, data_dir=DATA_DIR, save=True):
    """ Return the vocabulary (an instance of Vocab) made out of the ListOps validation file
    """

    df = pd.read_csv(data_dir + f'/val.tsv', sep='\t', usecols=['Source','Target'])
    #df.columns = ["label", "input1_id", "input2_id", "text1", "text2"]

    # decode to the right format
    df['Source'] = df['Source'].apply(listops_tokenizer)
    iterator = [itertools.chain(a) for a in df["Source"]]

    print("Building vocab...")

    vocab = utils.build_vocab(
        iterator,
        specials=["<pad>", "<unk>"] + (["<bos>"] if append_bos else []) + (["<eos>"] if append_eos else [])
    )
    vocab.set_default_index(vocab["<unk>"])

    if save:
        with open(data_dir + "/vocab.pkl", "wb") as f: pkl.dump(vocab, f)

    return vocab

def process_data(
    vocab,
    max_len=2000,
    append_bos=False,
    append_eos=True,
    kind='train',
    data_dir=DATA_DIR
):
    print(f"Processing {kind} data...")
    df = pd.read_csv(data_dir + f'/{kind}.tsv', sep='\t', usecols=['Source','Target'])
    max_len = max_len - int(append_bos) - int(append_eos)
    tokenize = lambda example: listops_tokenizer(example)[:max_len]
    df['Source'] = df['Source'].apply(tokenize)
    df['Target'] = df['Target'].astype(int)
    
    numericalize = lambda example: vocab(
        (["<bos>"] if append_bos else []) + example + (["<eos>"] if append_eos else []))
    df['Source'] = df['Source'].apply(numericalize)
    df['Source'] = df['Source'].apply(
        lambda x: utils.pad_sequence(
            x, max_len=(max_len+int(append_bos)+int(append_eos)), pad_val=vocab['<pad>']))
    
    inputs = torch.tensor(df['Source'], dtype=torch.int32)
    targets = torch.tensor(df['Target'], dtype=torch.int64)

    print("preprocessing done.")

    return inputs, targets



if __name__ == '__main__':
    ListOps()