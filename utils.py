import copy
from itertools import product
import os
import socket
from collections import Counter, OrderedDict

import pandas as pd
import numpy as np
from scipy import special as ss

import opt_einsum as oe
from einops import rearrange

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




def pad_sequence(seq, max_len, pad_val=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    if isinstance(seq, list):
        l = seq + [pad_val] * (max_len - len(seq))
    else:
        # l should be an array
        l = np.concatenate([seq, np.full((max_len - len(seq),), pad_val, dtype=seq.dtype)])
    return l



def unpack_batch(batch, torch_device):
    """ how to unpack the batch, depending on its content
    """
    if len(batch) == 2:
        batch_x, batch_y = batch
        batch_lengths = None
    elif len(batch) == 3:
        batch_x, batch_y, batch_lengths = batch
        batch_lengths = batch_lengths.to(torch_device)
    return batch_x, batch_y, batch_lengths


class Vocab():

    def __init__(self, vocab) -> None:
        super(Vocab, self).__init__()
        self.vocab = vocab
        self.default_index = -1

    def __len__(self) -> int:
        return len(self.vocab)
    
    def __getitem__(self, token: str) -> int:
        return self.vocab.get(token, self.default_index)

    def __call__(self, tokens):
        return self.forward(tokens)

    def __str__(self):
        return str(self.vocab)

    def forward(self, tokens):
        """ numericalize a list of tokens, return a list of indices """
        ret_list = []
        for char in tokens:
            ret_list.append(self[char])

        return ret_list

    def lookup_indices(self, tokens):
        """ numericalize a list of tokens, return a numpy array of indices """
        return np.fromiter(
            (self.vocab.get(t, self.default_index) for t in tokens),
            dtype=np.int32
        )

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








""" Cauchy kernel """


#try: # Try CUDA extension
#    from extensions.cauchy.cauchy import cauchy_mult
#    has_cauchy_extension = True
#except:
#    log.warn(
#        "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
#    )
#    has_cauchy_extension = False
_c2r = torch.view_as_real
_r2c = torch.view_as_complex



# version avec pykeops:

#try: # Try pykeops
#import pykeops
#from pykeops.torch import Genred
#has_pykeops = True
#def cauchy_conj(v, z, w):
#    """ Pykeops version """
#    expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
#    expr_denom = 'ComplexMult(z-w, z-Conj(w))'
#
#    cauchy_mult = Genred(
#        f'ComplexDivide({expr_num}, {expr_denom})',
#        # expr_num,
#        # expr_denom,
#        [
#            'v = Vj(2)',
#            'z = Vi(2)',
#            'w = Vj(2)',
#        ],
#        reduction_op='Sum',
#        axis=1,
#        dtype='float32' if v.dtype == torch.cfloat else 'float64',
#    )
#
#    v, z, w = _broadcast_dims(v, z, w)
#    v = _c2r(v)
#    z = _c2r(z)
#    w = _c2r(w)
#
#    #r = 2*cauchy_mult(v, z, w, backend='GPU')
#    r = 2*cauchy_mult(v, z, w)
#    return _r2c(r)

#def _broadcast_dims(*tensors):
#    max_dim = max([len(tensor.shape) for tensor in tensors])
#    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
#    return tensors


#def cauchy_conj(v, z, w):
#    v = _c2r(v)
#    z = _c2r(z)
#    w = _c2r(w)
#
#    z_i = z[:, None, :]
#    w_j = w[None, :, :]
#    v_j = v[None, :, :]
#
#    diff = z_i - w_j
#    denom = diff * diff.conj()
#
#    numer = z_i * v_j - (v * w).sum(dim=0, keepdim=True)[None, :, :]
#
#    r = (numer / denom).sum(dim=1)
#
#    return _r2c(2 * r)


def cauchy_conj(v, z, w):
    z = z.unsqueeze(-1)
    v = v.unsqueeze(-2)
    w = w.unsqueeze(-2)
    r = (z*v.real - (v*w.conj()).real) / ((z-w.real)**2 + w.imag**2)
    # r =  ((z-w.real)**2 + w.imag**2)
    return 2 * torch.sum(r, dim=-1)





""" S4 HiPPO kernel utilities """

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)

def embed_c2r(A):
    #A = rearrange(A, '... m n -> ... m () n ()')
    A = A[..., :, None, :, None]
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    A = A.reshape(A.shape[0] * A.shape[1], A.shape[2] * A.shape[3])
    #return rearrange(A, 'm x n y -> (m x) (n y)')
    return A

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * torch.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = torch.ones(N, dtype=dtype) # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0) # (2 N)
    else: raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0) # (rank N)
    return P

def nplr(measure, N, rank=1, dtype=torch.float):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.cfloat
    if measure == 'random':
        dtype = torch.cfloat if dtype == torch.float else torch.cdouble
        # w = torch.randn(N//2, dtype=dtype)
        w = -torch.exp(torch.randn(N//2)) + 1j*torch.randn(N//2)
        P = torch.randn(rank, N//2, dtype=dtype)
        B = torch.randn(N//2, dtype=dtype)
        V = torch.eye(N, dtype=dtype)[..., :N//2] # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    V = V[..., 0::2].contiguous()

    V_inv = V.conj().transpose(-1, -2)

    B = oe.contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    P = oe.contract('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P


    return w, P, B, V



def main():
    v = _r2c(torch.randn(10,2))
    z = _r2c(torch.randn(10,2))
    w = _r2c(torch.randn(10,2))
    cauchy_conj(v,z,w)

if __name__ == "__main__":
    main()