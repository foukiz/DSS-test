import torch
import torch.nn as nn

import math

import opt_einsum as oe

from utils import reciprocal, hippo_skew_evals




class DSSKernel(nn.Module):
    # TODO reprendre https://github.com/ag1988/dss/blob/main/src/models/sequence/ss/standalone/dss.py
    # la class DSSKernel qui compute au choix l'une des deux versions de la paramétrisations
    # de la proposition 1 de https://arxiv.org/pdf/2203.14343

    def __init__(
        self,
        H,
        N=64,
        dt_min=1e-3,
        dt_max=1e-1,
        sep_dt_re_im=True,      # use separate deltas for real, imag parts of Lambda
        Lambda_init='hippo_skew_pos_imag',
        epsilon=1e-7,           # avoids division by 0
        version='exp',      # DSS implementation to use
    ):
        super().__init__()
        assert version in ['exp', 'softmax']
        self.version = version

        self.H = H
        self.N = N
        self.epsilon = epsilon
        self.sep_dt_re_im = sep_dt_re_im
        self.Lambda_init = Lambda_init

        # complex tensors are stored as real with an extra last dim of size 2 
        # to denote real, imag parts as ADAM moments are non-linear  
        log_dt, Lambda, W = self.init(N, H, dt_min, dt_max, Lambda_init)  # [H], [N 2], [H N 2]

        self.register_parameter('log_dt', torch.nn.Parameter(log_dt))

        if 'exp' in version:
            assert (Lambda[:,0] <= 0).all()
            self.register_parameter('Lambda_log_neg_re', torch.nn.Parameter((-Lambda[:,0]).log()))
            if 'im' in version:
                self.register_parameter('Lambda_log_im', torch.nn.Parameter(Lambda[:,1].log()))
            else:
                self.register_parameter('Lambda_im', torch.nn.Parameter(Lambda[:,1]))
        else:
            self.register_parameter('Lambda', torch.nn.Parameter(Lambda))  # [N,2]

        self.register_parameter('W', torch.nn.Parameter(W))      # [C H N]

    def init(self, N, H, dt_min, dt_max, Lambda_init):
        if Lambda_init == 'hippo_skew_pos_imag':
            w = hippo_skew_evals(2*N)[:N] - .5                          # [N]
        elif Lambda_init == 'randn':
            w = torch.randn(N, dtype=torch.cfloat)                      # [N]
        else:
            raise NotImplementedError(f"Lambda init {Lambda_init} is not implemented")

        Lambda = torch.view_as_real(w.reshape(-1).to(torch.cfloat))                   # [N 2]

        # log delta
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))  # [H]
        if self.sep_dt_re_im:
            log_dt = log_dt.view(-1,1).tile(2)                          # [H 2]

        W = torch.randn(H, N, 2)                              # [H N 2]
        return log_dt, Lambda, W            # Delta (discretization scale),
                                            # Lambda (singular values of A),
                                            # W (C . B vector)

    def _Lambda(self):
        if 'exp' in self.version:
            if 'im' in self.version:
                return -self.Lambda_log_neg_re.exp() + 1j*self.Lambda_log_im.exp()        # [N]
            return -self.Lambda_log_neg_re.exp() + 1j*self.Lambda_im                      # [N]
        if 'clip' in self.version:
            return self.Lambda[:,0].clip(max=self.max_real_Lambda) + 1j*self.Lambda[:,1]  # [N]
        return torch.view_as_complex(self.Lambda)

    def forward(self, L, state=None):
        assert L >= 1

        Lambda = self._Lambda()                                              # [N]
        W = torch.view_as_complex(self.W)                                   # [H N]

        # Delta * Lambda
        if self.sep_dt_re_im:
            # Lambda.real * dt0  +  1j * Lambda.imag * dt1
            dt_Lambda = torch.view_as_complex(
                self.log_dt.exp().unsqueeze(1) * torch.view_as_real(Lambda).unsqueeze(0)
            )                 # [H N]
        else:
            dt_Lambda = self.log_dt.exp().unsqueeze(-1) * Lambda             # [H N]

        P = dt_Lambda.unsqueeze(-1) * torch.arange(L, device=W.device)       # [H N L]
        # replace the sequence length in first dimension
        P = P.permute(-1, *range(P.ndim - 1))                               # [L H N]

        if self.version in ['softmax']:
            # fast softmax using structure of P
            # see Appendix A.2 in https://arxiv.org/abs/2203.14343
            Lambda_gt_0 = Lambda.real > 0                                    # [N]
            if Lambda_gt_0.any():
                with torch.no_grad():
                    P_max = dt_Lambda * (Lambda_gt_0 * (L-1))                # [H N]
                P = P - P_max.unsqueeze(0)                                  # [L H N]
            S = P.exp()                                                      # [L H N]

            dt_Lambda_neg = dt_Lambda * (1 - 2*Lambda_gt_0)                  # [H N]
            # 1 / S.sum(-1) == num / den
            num = dt_Lambda_neg.exp() - 1                                    # [H N]
            den = (dt_Lambda_neg * L).exp() - 1                              # [H N]
            W = W * num * reciprocal(den * Lambda, self.epsilon)             # [H N]
        else:
            S = P.exp()                                                      # [L H N]
            if 'no-scale' not in self.version:
                W = W * (dt_Lambda.exp() - 1.) * reciprocal(Lambda, clamp=True)  # [H N]

        return oe.contract('hn,lhn->lh', W, S).real.to(torch.float32), state     # [L H]



class DSSLayer(nn.Module):

    def __init__(
        self,
        input_size,
        state_size,
        bidirectional=False,
        bias=True,
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
        self.kernel = DSSKernel(self.h, self.n)
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