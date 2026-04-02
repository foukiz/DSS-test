import os
import sys

import torch
from torch.utils.data import TensorDataset
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
        super().__init__(
            train_size=25000, val_size=12500, test_size=12500, seq_length=max_len, **kwargs)

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

        if os.path.exists(self.data_dir + '/vocab.pkl'):
            with open(self.data_dir + '/vocab.pkl', "rb") as f: self.vocab = pkl.load(f)
        if not os.path.exists(self.data_dir + '/data.pkl'):
            if self.vocab is None:
                self.vocab = make_vocab()
            x_train, y_train = process_raw_data(self.vocab, max_len=self.seq_length, kind='train')
            x_val, y_val = process_raw_data(self.vocab, max_len=self.seq_length, kind='val')
            x_test, y_test = process_raw_data(self.vocab, max_len=self.seq_length, kind='test')
            data_dic = {
                'train_ds': (x_train, y_train),
                'val_ds': (x_val, y_val),
                'test_ds': (x_test, y_test),
                'vocab': self.vocab
            }
            with open(self.data_dir + "/data.pkl", "wb") as f:
                pkl.dump(data_dic, f)
        with open(self.data_dir + '/data.pkl', 'rb') as f:
            data_dic = pkl.load(f)
            self.vocab = data_dic['vocab']

        self.input_dimension = len(self.vocab)
        self.padding_idx = self.vocab["<pad>"]

        train_ds = TensorDataset(*data_dic['train_ds'])
        val_ds = TensorDataset(*data_dic['val_ds'])
        test_ds = TensorDataset(*data_dic['test_ds'])

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
        #df['text1'] + df['text2'],
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
    data_dir=DATA_DIR
):
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

        #return np.fromiter(
        #    #(vocab[token] for token in tokens),
        #    vocab[tokens],
        #    dtype=np.int32
        #)

    # turn to list and crop to desired max length
    #tokenize = lambda example: list(example)[:max_len]
    #df['text1'] = df['text1'].apply(tokenize)
    #df['text2'] = df['text2'].apply(tokenize)
#
    #print("Tokenized.")

    # map to numerical indexes and add bos/eos tokens if desired
    #numericalize = lambda example: vocab(
    #    (["<bos>"] if append_bos else []) + example + (["<eos>"] if append_eos else [])
    #)

    #def numericalize(example):
    #    tokens = itertools.chain(
    #        (["<bos>"] if append_bos else []), example, (["<eos>"] if append_eos else [])
    #    )
    #    return np.fromiter(
    #        vocab[tokens], dtype=np.int32
    #    )
    #df['text1'] = df['text1'].apply(numericalize)
    #df['text2'] = df['text2'].apply(numericalize)
    text1 = [numericalize(tokens) for tokens in df['text1']]
    text1 = np.stack(text1, axis=0)
    text2 = [numericalize(tokens) for tokens in df['text2']]
    text2 = np.stack(text2, axis=0)
    
    text = np.concatenate([text1, text2], axis=1)

    # padd to max length
    #padding = lambda x: utils.pad_sequence(x, max_len=(max_len+int(append_bos)+int(append_eos)), pad_val=vocab['<pad>'])
    #df['text1'] = df['text1'].apply(padding)
    #df['text2'] = df['text2'].apply(padding)

    # concatenate the two texts next to each other
    #df['text'] = df['text1'] + df['text2']
    #df.drop(columns=['text1', 'text2'], inplace=True)

    #inputs = torch.tensor(df['text'], dtype=torch.int32)
    inputs = torch.from_numpy(text)
    targets = torch.tensor(df['label'], dtype=torch.int64)

    print("preprocessing done.")

    return inputs, targets






if __name__ == "__main__":
    #with open(DATA_DIR + "/vocab.pkl", "rb") as f: vocab = pkl.load(f)
    #process_raw_data(vocab, kind='test_sample')
    AAN()

