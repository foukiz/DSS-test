import torch
import torch.nn as nn

import numpy as np

import opt_einsum as oe

import torch.nn.functional as F

from einops import rearrange
from flax.linen.initializers import lecun_normal

from kernels import (
    DSSKernel,
    GammaExpectationKernel,
    GammaMGFKernel,
    UniformExpectationKernel,
    ExponentialExpectationKernel,
    HippoSSKernel,
    S4DKernel,
    GammaExpectationComplexKernel,
    GammaExpectationComplexKernelNegRe
)

from utils import init_VinvB, Activation




class DSSLayer(nn.Module):

    VERSIONS = ['exp', 'softmax', 'mgf', 'gamma', 'gamma_mgf', 'uniform', 'exponential', 'gamma_complex', 'gamma_complex_neg_re']

    def __init__(
        self,
        input_size,
        state_size,
        bias=True,
        activation=None,
        dropout=0.0,
        version='exp',
        bidirectional=False,
        seed=None,
        max_kernel_length=None,  # max len of SSM kernel to be used
        **kwargs
    ):  
        assert version in self.VERSIONS, "version must be one of {}".format(self.VERSIONS)
        #if seed: torch.manual_seed(seed)
        super().__init__()

        self.h = input_size
        self.n = state_size
        self.bidirectional = bidirectional

        channels = 2 if self.bidirectional else 1

        self.D = nn.Parameter(torch.randn(self.h))
        
        self.max_kernel_length = max_kernel_length

        if version in ('exp', 'softmax', 'mgf'):
            self.kernel = DSSKernel(self.h, self.n, version=version, channels=channels)
        elif version == 'gamma':
            self.kernel = GammaExpectationKernel(self.h, channels=channels, **kwargs)
        elif version == 'uniform':
            self.kernel = UniformExpectationKernel(self.h, channels=channels, **kwargs)
        elif version == 'exponential':
            self.kernel = ExponentialExpectationKernel(self.h, channels=channels, **kwargs)
        elif version == 'gamma_mgf':
            self.kernel = GammaMGFKernel(self.h, channels=channels, **kwargs)
        elif version == 'gamma_complex':
            self.kernel = GammaExpectationComplexKernel(self.h, channels=channels, **kwargs)
        elif version == 'gamma_complex_neg_re':
            self.kernel = GammaExpectationComplexKernelNegRe(self.h, channels=channels, **kwargs)
        #self.bias = bias

        # should have been instantiated already
        self.activation = activation
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.output = nn.Linear(input_size, input_size, bias=bias)

    def forward(self, u): # absorbs return_output and transformer src mask
        """
        u: (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        # L (sequence length) is the second dimension, the first is the batch size
        L = u.size(-1)

        # Compute SS Kernel
        Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)
        k = self.kernel(L=Lk)  # (C H Lk)

        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = (F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))).squeeze(0)

        # y = multiply_polynomials(u.unsqueeze(1), k.unsqueeze(0))[..., :L]  # (B 1 H L), (1 H Lk) -> (B H L)
        n = L + Lk
        k_f = torch.fft.rfft(k, n=n)  # (H ~n/2)
        u_f = torch.fft.rfft(u, n=n)  # (B H ~n/2)
        y_f = oe.contract('bhl,hl->bhl', u_f, k_f)            # (B H ~n/2)
        y = torch.fft.irfft(y_f, n=n)[... ,:L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + oe.contract('bhl,h->bhl', u, self.D)
        #y = y + u * self.D[None,None,:]  # (B H L)

        y = self.dropout(self.activation(y))  # (B H L)

        y = self.output(y.transpose(-1, -2)).transpose(-1, -2)  # (B H L)

        return y        # (B H L)

    def compute_gradients(self, reduction='mean'):
        grads = {}
        k = self.kernel
        for name, t in k.named_parameters():
            if t.grad is None: continue
            grad = t.grad.detach().cpu().numpy()
            if reduction == 'mean':
                grads[name] = np.mean(np.abs(grad))
            elif reduction == 'max':
                grads[name] = np.max(np.abs(grad))
            else:
                raise ValueError("reduction {} is unknown ; valid options are 'mean', 'max'".format(reduction))
        return grads

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h



class S4Layer(DSSLayer):

    def __init__(
            self,
            input_size,
            state_size=64,
            bias=True,
            bidirectional=False,
            #l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            #activation='gelu', # activation in between SS and FF
            #initializer=None, # initializer on FF
            #weight_norm=False, # weight normalization on FF
            **kwargs,
        ):
        """ Implémentation de S4, basée sur celle de DSS: les calculs et les arguments sont
            essentiellement les mêmes, mis à part pour l
        """

        super().__init__(
            input_size,
            state_size,
            bias=bias,
            version='exp',
            bidirectional=bidirectional,
            **kwargs
        )

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=None, **kwargs)



class S4DLayer(nn.Module):

    def __init__(
        self,
        input_size,
        state_size,
        channels=1,
        activation='gelu',
        gate=None,
        gate_activation=None,
        mult_activation=None,
        final_activation='glu',
        postactivation=None,
        initializer=None,
        bidirectional=True,
        weight_norm=False,
        dropout=0.0,
        drop_kernel=0.0,
        tie_dropout=False,
        transposed=True,
        **kwargs
    ):
        """ Implémentation de S4, basée sur celle de DSS: les calculs et les arguments sont
            essentiellement les mêmes, mis à part pour l
        """

        super().__init__()

        self.h = input_size
        self.n = state_size
        self.channels = channels
        #self.transposed = transposed

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.activation = activation
        self.final_activation = Activation(activation=final_activation, dim=-2)

        d_input = self.h
        d_output = 2*d_input if final_activation == 'glu' else d_input
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_input, d_output, kernel_size=1),
            self.final_activation
        )

        self.D = nn.Parameter(torch.randn(self.h))

        self.bidirectional = bidirectional
        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kwargs)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

    def forward(self, u): # absorbs return_output and transformer src mask
        """
        u: (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        # L (sequence length) is the second dimension, the first is the batch size
        L = u.size(-1)

        # Compute SS Kernel
        #Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)
        Lk = L
        k = self.kernel(L=Lk)  # (H Lk)

        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = (F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))).squeeze(0)

        k = self.drop_kernel(k)

        # y = multiply_polynomials(u.unsqueeze(1), k.unsqueeze(0))[..., :L]  # (B 1 H L), (1 H Lk) -> (B H L)
        n = L + Lk
        k_f = torch.fft.rfft(k, n=n)  # (H ~n/2)
        u_f = torch.fft.rfft(u, n=n)  # (B H ~n/2)
        y_f = oe.contract('bhl,hl->bhl', u_f, k_f)            # (B H ~n/2)
        y = torch.fft.irfft(y_f, n=n)[... ,:L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + oe.contract('bhl,h->bhl', u, self.D)
        #y = y + u * self.D[None,None,:]  # (B H L)

        y = self.drop(self.activation(self.drop(y)))  # (B H L)

        #y = self.output_linear(y.transpose(-1, -2)).transpose(-1, -2)  # (B H L)
        y = self.output_linear(y)                                 # (B H L)

        return y        # (B H L)



class S5Layer(nn.Module):

    def __init__(
        self,
        input_size,
        state_size,
        Lambda_re_init,
        Lambda_im_init,
        V,
        Vinv,
        C_init,
        discretization,
        dt_min,
        dt_max,
        conj_sym=True,
        clip_eigs=False,
        bidirectional=False,
        step_rescale=1.0
    ):
        super().__init__()
        
         # ---------- Hyperparamètres ----------
        self.H = input_size
        self.P = state_size
        self.C_init = C_init
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale

        # ---------- Constantes ----------
        self.register_buffer("V", V)
        self.register_buffer("Vinv", Vinv)

        # ---------- Paramètres ----------
        self.Lambda_re = nn.Parameter(Lambda_re_init.clone())
        self.Lambda_im = nn.Parameter(Lambda_im_init.clone())

        if conj_sym:
            local_P = 2 * state_size
        else:
            local_P = state_size
        
        B_init = lecun_normal()
        B_shape = (local_P, self.H)

        B = init_VinvB(B_init, rng, self.H, self.Vinv)
        self.B = nn.Parameter(B)


    def forward(self, u):
        raise NotImplementedError("S5Layer is not implemented yet")





class InputEncoder(nn.Module):
    # TODO une classe d'encoder d'input qui met les données sous le bon format pour le DSSLayer
    # par exemple un embedding pour ListOps, ou une simple couche linéaire pour CopyTask

    def __init__(self, data_dim, input_size, mode='embedding', **kwargs):
        super().__init__()
        assert mode in ['embedding', 'linear', 'identity'], (f"mode must be one of "
                                "['embedding', 'linear', 'identity'], found {mode}")
        self.data_dim = data_dim
        self.input_size = input_size
        self.mode = mode

        if mode == 'embedding':
            if "padding_idx" in kwargs:
                padding_idx = kwargs['padding_idx']
            else:
                padding_idx = None
            self.layer = nn.Embedding(data_dim, input_size, padding_idx=padding_idx)
        if mode == 'linear':
            self.layer = nn.Linear(data_dim, input_size)
        if mode == 'identity':
            assert data_dim == input_size, ("for identity encoding, input_dim "
                                             "must be equal to input_size")
            self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x)




class TopPooling(nn.Module):
    """ A layer to put on top of the sequence model that outputs a sequence to extract
        a single vector out of the sequence, or the whole sequence.
    """

    def __init__(self, mode='last', use_lengths=False):
        super().__init__()
        assert mode in ['average', 'last', 'manytomany'], "mode must be one of ['average', 'last', 'manytomany']"
        self.mode = mode
        self.use_lengths = use_lengths

    def forward(self, x, batch_lengths=None):
        """ Sequence should be of shape (B, L, N)
        """

        if self.mode == 'average':
            restrict = lambda x: x.mean(dim=-2)  # (B, N)
        elif self.mode == 'last':
            restrict = lambda x: x[..., -1, :]   # (B, N)
        elif self.mode == 'manytomany':
            restrict = lambda x: x  # (B, L, N)
        else:
            raise NotImplementedError(f"Pooling mode {self.mode} not implemented")
        
        if self.use_lengths:
            assert batch_lengths is not None
            x = torch.stack([
                restrict(out[..., :length, :])
                for out, length
                in zip(torch.unbind(x, dim=0), batch_lengths)
            ], dim=0)
        else:
            x = restrict(x)

        return x
    


class RetrievalHead(nn.Module):

    def __init__(self, input_size, output_size, activation='gelu'):
        super().__init__()
        if activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented for RetrievalHead")
        
        self.classifier = nn.Sequential(
            nn.Linear(4*input_size, input_size),
            activation_fn,
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        """ x : (2*B, H)
        """
        x = rearrange(x, '(z b) d -> z b d', z=2)
        x1, x2 = x[0], x[1]
        features = torch.cat([x1, x2, x1-x2, x1*x2], dim=-1)
        logits = self.classifier(features)

        return logits



class Normalization(nn.Module):

    def __init__(self, input_size, mode='batch_norm'):
        super().__init__()
        self.input_size = input_size
        self.mode = mode if mode is not None else 'none'

        assert mode in ['batch_norm', 'layer_norm', 'none'], "mode must be one of ['batch_norm', 'layer_norm', 'none']"

        if mode == 'batch_norm':
            #self.norm = TransposeBatchNorm(input_size)
            self.norm = nn.BatchNorm1d(input_size)
        elif mode == 'layer_norm':
            self.norm = CustomLayerNorm(input_size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        #if transpose: x = x.transpose(-1, -2)  # (B, H, L) -> (B, L, H)
        x = self.norm(x)
        #if transpose: x = x.transpose(-1, -2)  # (B, L, H) -> (B, H, L)

        return x



class TransposeBatchNorm(nn.Module):
    """ A classic batch norm layer, but the input is expected to have shape (B, L, H)
        and the normalization is performed on the H dimension
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        # x shape is (B, L, H)
        x = x.transpose(-1, -2)  # (B, L, H) -> (B, H, L)
        x = self.norm(x)
        x = x.transpose(-1, -2)  # (B, H, L)-> (B, L, H)
        return x


class CustomLayerNorm(nn.Module):
    """ expects shape (B, H, L)
    """

    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            s, m = torch.std_mean(x, dim=-2, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            y = self.ln(x.transpose(-1,-2)).transpose(-1,-2)
        return y


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        #self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            #if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            #if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X




if __name__ == "__main__":
    h = 4
    N = 8
    L = 16
    B = 5
    C = 2
    layer = S4Layer(input_size=h, state_size=N)
    u = torch.randn(B, L, h)
    layer(u)