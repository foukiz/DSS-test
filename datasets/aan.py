import os
import sys

import torch
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np
import pandas as pd
import pickle as pkl
import ast
import itertools

try:
    from .dataset import Dataset
except ImportError:
    from dataset import Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import utils



DATA_DIR = 'data/aan_data'

TRAIN_SIZE = 147085
VAL_SIZE = 18089
TEST_SIZE = 17436




class AAN(Dataset):

    def __init__(
        self,
        max_len=4000,
        data_dir=DATA_DIR,
        **kwargs
    ):
        """ Class to generate the AAN (ACL Anthology Network) dataset with some properties
            This dataset is used for the 'Retrieval' benchmark in LRA
        """

        self.data_dir = data_dir
        self.vocab = None
        self.seq_length = max_len
        train_size = kwargs.pop('train_size', TRAIN_SIZE)
        val_size = kwargs.pop('val_size', VAL_SIZE)
        test_size = kwargs.pop('test_size', TEST_SIZE)
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
    
    def make_train_ds(self, train_files):
        datasets = []
        for f in train_files:
            ds = torch.load(f)
            datasets.append(ds)
        train_ds = ConcatDataset(datasets)
        
        return train_ds

    def import_dataset(self):

        print("-" * 60 + f"Loading {type(self).__name__}" + "-" * 60)

        if os.path.exists(self.data_dir + '/vocab.pkl'):
            with open(self.data_dir + '/vocab.pkl', "rb") as f: self.vocab = pkl.load(f)
        else:
            self.vocab = make_vocab()
        if not os.path.exists(self.data_dir + '/aan_train_0.pt'):
            process_raw_data(self.vocab, max_len=self.seq_length, kind='train')
        if not os.path.exists(self.data_dir + '/aan_val.pt'):
            process_raw_data(self.vocab, max_len=self.seq_length, kind='val')
        if not os.path.exists(self.data_dir + '/aan_test.pt'):
            process_raw_data(self.vocab, max_len=self.seq_length, kind='test')

        self.input_dimension = len(self.vocab)
        self.padding_idx = self.vocab["<pad>"]

        data_files = os.listdir(self.data_dir)
        train_files = [f'{self.data_dir}/{f}' for f in data_files if f.startswith('aan_train')]

        train_ds = self.make_train_ds(train_files)
        val_ds = torch.load(f'{self.data_dir}/aan_val.pt')
        test_ds = torch.load(f'{self.data_dir}/aan_test.pt')
        #val_ds = TensorDataset(*torch.load(f'{self.data_dir}/aan_val.pt'))
        #test_ds = TensorDataset(*torch.load(f'{self.data_dir}/aan_test.pt'))

        print("-" * 60 + f"{type(self).__name__} loaded" + "-" * 60)

        return train_ds, val_ds, test_ds




def make_vocab(append_bos=False, append_eos=True, data_dir=DATA_DIR, save=True):
    """ Return the vocabulary (an instance of Vocab) made out of the AAN train file
    """
    
    df = pd.read_csv(
        data_dir + f'/new_aan_pairs.train.tsv',
        sep="\t"
    )
    df = df.dropna()
    df.columns = ["label", "input1_id", "input2_id", "text1", "text2"]
    
    # decode to the right format
    decode = lambda x: ast.literal_eval(x).decode("utf-8") 
    df['text1'] = df['text1'].apply(decode)
    df['text2'] = df['text2'].apply(decode)
    text = [itertools.chain(a, b) for a, b in zip(df["text1"], df["text2"])]

    print("Building vocab...")

    vocab = utils.build_vocab(
        text,
        specials=["<pad>", "<unk>"] + (["<bos>"] if append_bos else []) + (["<eos>"] if append_eos else [])
    )
    vocab.set_default_index(vocab["<unk>"])

    if save:
        with open(data_dir + "/vocab.pkl", "wb") as f: pkl.dump(vocab, f)

    return vocab


def process_raw_data(
    vocab,
    max_len=4000,
    append_bos=False,
    append_eos=True,
    kind='train',
    data_dir=DATA_DIR,
    file_size=6400,
):
    """ process raw data and save it as tensors in the data directory, with the
        right format for the AAN dataset. The train set is split into multiple files
        to avoid memory issues, while the val and test sets are saved in a single file each.
    """
    
    print(f"Processing {kind} data...")
    max_len = max_len - int(append_bos) - int(append_eos)
    df = pd.read_csv(
        data_dir + f'/new_aan_pairs.{kind}.tsv',
        sep="\t"
    )
    df = df.dropna()
    df.columns = ["label", "input1_id", "input2_id", "text1", "text2"]
    df.drop(columns=['input1_id', 'input2_id'], inplace=True)
    df['label'] = df['label'].astype('int64')

    decode = lambda x: ast.literal_eval(x).decode("utf-8") 
    df['text1'] = df['text1'].apply(decode)
    df['text2'] = df['text2'].apply(decode)
    targets = torch.tensor(df['label'], dtype=torch.int64)

    def numericalize(example):
        tokens = itertools.chain(
            ["<bos>"] if append_bos else [],
            itertools.islice(example, max_len),
            ["<eos>"] if append_eos else []
        )
        indices = vocab.lookup_indices(tokens)
        if len(indices) < max_len + int(append_bos) + int(append_eos):
            indices = utils.pad_sequence(indices, max_len + int(append_bos) + int(append_eos), pad_val=vocab["<pad>"])

        return indices

    if kind == 'train':
        num_files = (len(df) + file_size - 1) // file_size
        for i in range(num_files):
            start_idx = i * file_size
            end_idx = min((i + 1) * file_size, len(df))
            df_chunk = df.iloc[start_idx:end_idx]

            text1 = [numericalize(tokens) for tokens in df_chunk['text1']]
            text1 = np.stack(text1, axis=0)
            text2 = [numericalize(tokens) for tokens in df_chunk['text2']]
            text2 = np.stack(text2, axis=0)
            text = torch.from_numpy(np.concatenate([text1, text2], axis=1))
            labels = targets[start_idx:end_idx]
            ds = TensorDataset(text, labels)
            torch.save(ds, f'{data_dir}/aan_{kind}_{i}.pt')
        print("preprocessing done.")

    else:
        text1 = [numericalize(tokens) for tokens in df['text1']]
        text1 = np.stack(text1, axis=0)
        text2 = [numericalize(tokens) for tokens in df['text2']]
        text2 = np.stack(text2, axis=0)
        text = torch.from_numpy(np.concatenate([text1, text2], axis=1))
        ds = TensorDataset(text, targets)
        torch.save(ds, f'{data_dir}/aan_{kind}.pt')

        print("preprocessing done.")







if __name__ == "__main__":
    AAN()

