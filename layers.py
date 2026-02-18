import torch
import torch.nn as nn

import opt_einsum as oe

from kernel import DSSKernel





class DSSLayer(nn.Module):

    def __init__(
        self,
        input_size,
        state_size,
        bias=True,
        version='exp',
        bidirectional=False,
        seed=None,
        max_kernel_length=None,  # max len of SSM kernel to be used
        **kwargs
    ):
        if seed: torch.manual_seed(seed)
        super().__init__()

        self.h = input_size
        self.n = state_size
        self.bidirectional = bidirectional

        self.D = nn.Parameter(torch.randn(self.h))
        
        self.max_kernel_length = max_kernel_length
        self.kernel = DSSKernel(self.h, self.n, version=version)
        self.bias = bias

    def forward(self, u): # absorbs return_output and transformer src mask
        """
        u: (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        # L (sequence length) is the second dimension, the first is the batch size
        L = u.size(1)

        # Compute SS Kernel
        Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)
        k, _ = self.kernel(L=Lk)  # (Lk H) (Lk B H)
        
        # y = multiply_polynomials(u.unsqueeze(1), k.unsqueeze(0))[..., :L]  # (B 1 H L), (1 H Lk) -> (B H L)
        # fft has to be performed along the seuquence length dimension ;
        # hence the arguments dim=0 and dim=1 respectively in k_f and u_f
        n = L + Lk
        k_f = torch.fft.rfft(k, dim=0, n=n)  # (~n/2 H)
        u_f = torch.fft.rfft(u, dim=1, n=n)  # (B ~n/2 H)
        y_f = k_f[None,:,:] * u_f
        y = torch.fft.irfft(y_f, dim=1, n=n)[:,:L,:] # (B L H)

        # Compute D term in state space equation - essentially a skip connection
        #y = y + contract('bhl,h->bhl', u, self.D)
        y = y + u * self.D[None,None,:]  # (B H L)

        return y

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h
    



class InputEncoder(nn.Module):
    # TODO une classe d'encoder d'input qui met les données sous le bon format pour le DSSLayer
    # par exemple un embedding pour ListOps, ou une simple couche linéaire pour CopyTask

    def __init__(self, data_dim, input_size, mode='embedding'):
        super().__init__()
        assert mode in ['embedding', 'linear', 'identity'], (f"mode must be one of "
                                "['embedding', 'linear', 'identity'], found {mode}")
        self.data_dim = data_dim
        self.input_size = input_size
        self.mode = mode

        if mode == 'embedding':
            self.layer = nn.Embedding(data_dim, input_size)
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

    def __init__(self, mode='last'):
        super().__init__()
        assert mode in ['average', 'last', 'manytomany'], "mode must be one of ['average', 'last', 'manytomany']"
        self.mode = mode

    def forward(self, x):
        """ Sequence should be of shape (B, L, N)
        """

        if self.mode == 'average':
            x = x.mean(dim=-2)  # (B, N)
        elif self.mode == 'last':
            x = x[:, -1, :]   # (B, N)
        elif self.mode == 'manytomany':
            pass  # (B, L, N)
        else:
            raise NotImplementedError(f"Pooling mode {self.mode} not implemented")
        return x


class Normalization(nn.Module):

    def __init__(self, input_size, mode='batch_norm'):
        super().__init__()
        self.input_size = input_size
        self.mode = mode if mode is not None else 'none'

        assert mode in ['batch_norm', 'layer_norm', 'none'], "mode must be one of ['batch_norm', 'layer_norm', 'none']"

        if mode == 'batch_norm':
            self.norm = nn.BatchNorm1d(input_size)
        elif mode == 'layer_norm':
            self.norm = nn.LayerNorm(input_size)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        # x shape is (B, L, H)
        B, L, H = x.shape
        x = x.view(-1, H)  # (B*L, H)
        x = self.norm(x)
        x = x.view(B, L, H)  # (B, L, H)
        return x