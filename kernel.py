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
        assert version in ['exp', 'softmax', 'mgf']
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

        # Lambda en version complexe
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
    





class GammaExpectationKernel(nn.Module):
    """ Kernel computed as the expectation of (e^{Delta * X} - 1) / X * e^{Delta X j} where X
        is a Gamma distributed random variable with shape parameter alpha, and scale parameter
        theta.
    """
    
    def __init__(
            self,
            H,
            dt_min=1e-3,
            dt_max=1e-1,
            alpha_mean=5.0,
            alpha_std=1.0,
            theta_mean=1.0,
            theta_std=0.5
        ):
        assert alpha_mean > 0 and alpha_std >= 0 and theta_mean > 0 and theta_std >= 0, "alpha_mean, alpha_std, theta_mean, theta_std must be positive"
        super().__init__()
        self.H = H
        log_dt, log_alpha, log_theta = self.init(H, dt_min, dt_max, alpha_mean, alpha_std, theta_mean, theta_std)
        self.register_parameter('log_dt', torch.nn.Parameter(log_dt))
        self.register_parameter('log_alpha', torch.nn.Parameter(log_alpha))
        self.register_parameter('log_theta', torch.nn.Parameter(log_theta))

    def init(self, H, dt_min, dt_max, alpha_mean, alpha_std, theta_mean, theta_std):
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
        
        # alpha, theta are respectively the shape and scale parameters of the Gamma distribution
        while True:
            alpha = torch.randn(H) * alpha_std + alpha_mean
            theta = torch.randn(H) * theta_std + theta_mean
            if (alpha > 0).all() and (theta > 0).all():
                break
        return log_dt, alpha.log(), theta.log()

    def forward(self, L, state=None):
        Delta = self.log_dt.exp().unsqueeze(-1)                                   # [H]
        alpha = self.log_alpha.exp().unsqueeze(-1)                                # [H,1]
        theta = self.log_theta.exp().unsqueeze(-1)                                # [H,1]

        beta = 1. / theta + Delta * torch.arange(L+1, device=theta.device)        # [H L+1]
        k = (1. / beta[...,:-1]**(alpha-1) - 1. / beta[...,1:]) / ((alpha - 1) * theta**alpha) # [H L]

        return k
