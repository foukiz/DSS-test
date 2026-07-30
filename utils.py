import copy
from itertools import product
import os
import socket
from collections import Counter, OrderedDict

import math
import pandas as pd
import numpy as np
from scipy import special as ss

import opt_einsum as oe
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
import torch.nn as nn


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



def unpack_batch(batch, torch_device=None):
    """ how to unpack the batch, depending on its content
    """
    if len(batch) == 2:
        batch_x, batch_y = batch
        batch_lengths = None
    elif len(batch) == 3:
        batch_x, batch_y, batch_lengths = batch
        if torch_device is not None:
            batch_lengths = batch_lengths.to(torch_device)
    # le cas AAN, où on a batch_x1, batch_x2, batch_lengths, batch_ids, batch_texts
    elif len(batch) == 5:
        batch_x1, batch_x2, batch_y, batch_lengths1, batch_lengths2 = batch
        if torch_device is not None:
            batch_lengths1 = batch_lengths1.to(torch_device)
            batch_lengths2 = batch_lengths2.to(torch_device)
        batch_x = torch.cat([batch_x1, batch_x2], dim=0)
        batch_lengths = torch.cat([batch_lengths1, batch_lengths2], dim=0)
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



def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))



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

def nplr(measure, N, rank=1, dtype=torch.float, diagonalize_precision=True, B_clip=2.0):
    """Constructs NPLR form of HiPPO matrices.

    Returns w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B

    measure: Name of HiPPO method.
    N: Size of recurrent A matrix (also known as `d_state` elsewhere).
    dtype: Single or double precision.
    diagonalize_precision: Calculate diagonalization in double precision.
    B_clip: Clip values of B, can help with stability. None for no clipping.
    """

    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype) # (r N)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)

    # We require AP to be nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0,0]*torch.eye(N))**2) / N) > 1e-5: # if not torch.allclose(_A - _A[0,0]*torch.eye(N), torch.zeros(N, N), atol=1e-5):
        print("WARNING: HiPPO matrix not skew symmetric", err)


    # Take advantage of identity + skew-symmetric form to calculate real and imaginary parts separately
    # Imaginary part can use eigh instead of eig
    W_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision: AP = AP.to(torch.double)
    # w, V = torch.linalg.eig(AP) # (..., N) (..., N, N)
    W_im, V = torch.linalg.eigh(AP*-1j) # (..., N) (..., N, N)
    if diagonalize_precision: W_im, V = W_im.to(cdtype), V.to(cdtype)
    W = W_re + 1j * W_im
    # Check: V W V^{-1} = A
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))


    # Only keep half of each conjugate pair
    _, idx = torch.sort(W.imag)
    W_sorted = W[idx]
    V_sorted = V[:, idx]

    # There is an edge case when eigenvalues can be 0, which requires some machinery to handle
    # We use a huge hack here: Assume only one pair is 0, and that it is the first row/column of A (only happens in Fourier case)
    V = V_sorted[:, :N//2]
    W = W_sorted[:N//2]  # Only keep negative imaginary components
    assert W[-2].abs() > 1e-4, "Only 1 zero eigenvalue allowed in diagonal part of A"
    if W[-1].abs() < 1e-4:
        V[:, -1] = 0.
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2)
    if ((err := torch.sum((2*_AP.real-AP)**2)/N) > 1e-5):
        print("Warning: Diagonalization of A matrix not numerically precise - error", err)
    # print("check", V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2))

    V_inv = V.conj().transpose(-1, -2)

    # C = initial_C(measure, N, dtype=dtype)
    B = torch.einsum('ij, j -> i', V_inv, B.to(V)) # V^* B
    # C = contract('ij, j -> i', V_inv, C.to(V)) # V^* C
    P = torch.einsum('ij, ...j -> ...i', V_inv, P.to(V)) # V^* P

    if B_clip is not None:
        B = B.real + 1j*torch.clamp(B.imag, min=-B_clip, max=B_clip)

    # W represents the imaginary part of the DPLR form: A = W - PP^*
    # Downstream classes just call this A for simplicity,
    # which is also more consistent with the diagonal case
    return W, P, B, V

def dplr(
    init='hippo',
    N=64, rank=1, H=1,
    dtype=torch.float,
    real_random=False,
    real_scale=1.0,
    imag_random=False,
    imag_scale=1.0,
    B_init='constant',
    B_scale=1.0,
    P_scale=1.0,
    normalize=False,
):
    """Directly construct a DPLR matrix.

    Args:
    - init: (str) ['rand', 'lin', inv', 'real', 'hippo'] Choices for initialization of A.
          Most of these affect the imaginary part of A, except for 'real'.
    - real_random: (bool) Initialize A.real in -U[0, 1]. Otherwise, initialize to -1/2.
    - real_scale: (float) Scaling factor of real part of A.
    - imag_random: (bool) Initialize A.imag randomly.
    - imag_scale: (bool) Scaling factor of imaginary part of A.
    - B_init: (str) ['constant' | 'random' | 'alternating' | 'unit-cw' | 'unit-ccw' | 'hippo']
          Choices for initialization of B.
    - B_scale: (float) Scaling factor for B
    - P_scale: (float) Scaling factor for P
    - normalize: (bool) Apply an automatic normalization factor on B
    """
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)

    # Construct real part of diagonal A (must be non-negative)
    if real_random:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    real_part = real_scale * real_part

    # Construct imaginary part of diagonal A (must be non-negative)
    if imag_random:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    if init in ['random', 'rand']:
        imag_part = torch.exp(torch.randn(H, N//2))
    elif init == 'real':
        imag_part = 0 * imag_part
        if real_random:
            real_part = torch.rand(H, N//2) * N//2
        else:
            # This is the S4D-Real method described in the S4D paper
            # The A matrix is diag(-1, -2, ..., -N), which are the eigenvalues of the HiPPO matrix
            real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif init in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif init in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif init in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif init in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    elif init in ['legs', 'hippo']:
        A, _, _, _ = nplr('legs', N)
        imag_part = -A.imag  # Positive
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part

    # Construct diagonal A
    A = -real_part - 1j * imag_part  # Force negative real and imag
    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)  # Allow some tolerance for numerical precision on real part

    if init in ['legs', 'hippo']:
        # Special initialization using the HiPPO B matrix
        # Note that theory (from S4D paper) says that B should be halved
        # to match DPLR but we drop this 0.5 factor for simplicity
        _, P, B, _ = nplr('legs', N, B_clip=2.0)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'constant':
        B = torch.ones(H, N//2, dtype=dtype)
    elif B_init == 'random':
        B = torch.randn(H, N//2, dtype=dtype)
    elif B_init == 'alternating':  # Seems to track 'constant' exactly for some reason
        B = torch.ones(H, N//4, 2, dtype=dtype)
        B[:, :, 1] *= -1
        B = B.view(H, N//2)
    elif B_init == 'unit-cw':
        z = torch.tensor(torch.exp(-2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'unit-ccw':
        z = torch.tensor(torch.exp(2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    else: raise NotImplementedError
    B *= B_scale

    # Experimental feature that appeared in earlier versions of HTTYH (not extensively tested)
    # Seems more principled for normalization theoretically, but seemed to hurt on PathX
    if normalize:
        norm = -B/A # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    # Initialize P
    if B_init in ['legs', 'hippo']:
        # P constructed earlier
        P = repeat(P, 'r n -> r h n', h=H).clone().contiguous()
    else:
        P = torch.randn(rank, H, N//2, dtype=dtype)
        P = P * P_scale

    # Initialize V (only used in testing)
    V = torch.eye(N, dtype=dtype)[:, :N//2]
    V = repeat(V, 'n m -> h n m', h=H)

    return A, P, B, V




""" S4D utilities """

def inv_transform(param, transform='none'):
    """Initialize a (positive) parameter under a transform."""
    param = torch.clamp(param, min=1e-4)
    if transform == 'none':
        return param
    elif transform == 'exp':
        return torch.log(param) # Some of the HiPPO methods have real part 0
    elif transform == 'relu':
        return param
    elif transform == 'sigmoid':
        return torch.logit(param)
    elif transform == 'softplus':
        return torch.log(torch.exp(param)-1)
    else: raise NotImplementedError

def param_transform(param, transform='none'):
    """Get a (positive) parameter under a transform."""
    if transform == 'none':
        p = param
    elif transform == 'exp':
        p = torch.exp(param)
    elif transform == 'relu':
        # JAX version seems to NaN if you allow 0's, although this code was fine without it
        p = F.relu(param)+1e-4
    elif transform == 'sigmoid':
        p = F.sigmoid(param)
    elif transform == 'softplus':
        p = F.softplus(param)
    else: raise NotImplementedError
    return p

def log_vandermonde_naive(v, x, L, conj=True):
    """ v: (..., N)
        x: (..., N)
        returns: (..., L) \sum v x^l
    """

    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = torch.einsum('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    return 2*vandermonde_prod.real





""" S5 utilities """

def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)




def main():
    v = _r2c(torch.randn(10,2))
    z = _r2c(torch.randn(10,2))
    w = _r2c(torch.randn(10,2))
    cauchy_conj(v,z,w)

if __name__ == "__main__":
    main()