import copy
from itertools import product
import os
import socket
from collections import Counter, OrderedDict

import pandas as pd

import torch


PATH = os.getcwd()



def find_file(file_name):
    file_name = os.path.basename(file_name)
    for root, _, files in os.walk(PATH):
        if file_name in files:
            return os.path.join(root, file_name)
    raise FileNotFoundError("file {} do not exist".format(file_name))



def reciprocal(x, epsilon=1e-7, clamp=False):
    """ fancy inverse function with stability factor epsilon ;
        returns 1 / x, with bounded norm
    """
    # used to stabilise the softmax function applied to complex terms
    # see Appendix A.2 in https://arxiv.org/abs/2203.14343
    x_conj = x.conj()
    norm_sq = (x*x_conj).real.clamp(epsilon) if clamp else (x*x_conj + epsilon)
    return x_conj / norm_sq



def hippo_skew_evals(N):
    """ eigenvalues of (Hippo - Hippo.t()) / 2  (largest imag part first) """
    i = torch.arange(N, dtype=torch.float)
    x = 2*i + 1
    Hippo = (x.view(-1,1) * x.view(1,-1)).sqrt().tril(diagonal=-1)  # [N N]
    Skew = (Hippo - Hippo.t()) / 2                                  # [N N] 
    evals = torch.linalg.eigvals(Skew)                              # [N]
    # decreasing order of imag
    return evals[evals.imag.argsort(descending=True)]               # [N]





def extract_leaf_lists(d, parent_path=()):
    """ Retourne toutes les feuilles d'un dictionnaire imbriqué qui sont des listes,
        ainsi que leur path complet.

        args: d, type = dict
    """
    leaves = []
    for k, v in d.items():
        path = parent_path + (k,)
        if isinstance(v, dict):
            leaves.extend(extract_leaf_lists(v, path))
        elif isinstance(v, list):
            leaves.append((path, v))
    return leaves


def set_by_path(d, path, value):
    """ Modifie un dictionnaire nested à un chemin donné.
    """
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def iter_configs(config):
    """ Génère toutes les configurations possibles à partir d'un dict nested contenant des listes.
        Retourne un iterator (generator).
    """
    
    leaves = extract_leaf_lists(config)

    # Aucun hyperparamètre variable → yield config unique
    if not leaves:
        yield config
        return

    paths = [p for p, _ in leaves]
    values_list = [v for _, v in leaves]

    # Parcours cartésien, mais une config à la fois
    for combo in product(*values_list):
        cfg_copy = copy.deepcopy(config)
        for path, val in zip(paths, combo):
            set_by_path(cfg_copy, path, val)
        yield cfg_copy



def assert_single_run_config(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            assert_single_run_config(v, f"{path}.{k}" if path else k)
    else:
        assert not isinstance(d, list), f"Leaf at '{path}' is a list, config is not single-run formatted"




def has_internet(host="api.wandb.ai", port=443, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((host, port))
        return True
    except OSError:
        return False





class Vocab():

    def __init__(self, vocab) -> None:
        super(Vocab, self).__init__()
        self.vocab = vocab
        self.default_index = -1

    def __len__(self) -> int:
        return len(self.vocab)
    
    def __getitem__(self, token: str) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            return self.default_index
        
    def __call__(self, tokens):
        return self.forward(tokens)
    
    def __str__(self):
        return str(self.vocab)

    def forward(self, list_of_tokens):
        ret_list = []
        for char in list_of_tokens:
            ret_list.append(self[char])
        
        return ret_list

    def set_default_index(self, index):
        self.default_index = index


def build_vocab(texts, min_freq=1, specials=[], special_first=True):
    freqs = Counter()

    for text in texts:
        freqs.update(text)

    chars = []
    for token, freq in freqs.items():
        if freq >= min_freq:
            chars.append(token)
    
    if special_first:
        chars[0:0] = specials
    else:
        chars.extend(specials)

    vocab = {val: i for i, val in enumerate(chars)}

    return Vocab(vocab)


if __name__ == '__main__':
    texts = ["je m'appelle armand", "les caractères rares sont par exemple µ et @"]
    specials = ["<pad>", "<unk>", "<eos>"]
    vocab = build_vocab(texts, min_freq=2, specials=specials)
    vocab.set_default_index(vocab["<unk>"])
    print("vocab is")
    print(str(vocab)+'\n')
    test_sent = "test pour une phrase avec un càrct incônu"
    print(test_sent+'\n')
    print(vocab("test pour une phrase avec un càrct incônu"))
    print()