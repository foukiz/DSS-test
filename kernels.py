import torch
import torch.nn as nn
import numpy as np

import math

import opt_einsum as oe
from einops import rearrange, repeat

from utils import reciprocal, hippo_skew_evals, nplr, power, cauchy_conj



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
        version='exp',      # DSS version to use
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

        self.register_parameter('W', torch.nn.Parameter(W))      # [H N]

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

        # version papier DSS, qui retourne l'état caché (mais ne sert à rien)
        #return oe.contract('hn,lhn->lh', W, S).real.to(torch.float32), state     # [L H]
        return oe.contract('hn,lhn->lh', W, S).real.to(torch.float32)            # [L H]





class GammaExpectationKernel(nn.Module):
    """ Kernel computed as the expectation of (e^{Delta * X} - 1) / X * e^{Delta X j} where X
        is a Gamma distributed random variable with shape parameter alpha, and scale parameter
        theta.
    """
    
    def __init__(
            self,
            H,
            epsilon=1e-6,           # avoids division by 0
            dt_min=1e-3,
            dt_max=1e-1,
            alpha_mean=4.0,
            alpha_std=1.0,
            theta_mean=1.0,
            theta_std=0.5,
            order_2_approx=False,
            **kwargs
        ):
        assert alpha_mean > 0 and alpha_std >= 0 and theta_mean > 0 and theta_std >= 0, "alpha_mean, alpha_std, theta_mean, theta_std must be positive and alpha_std >= 0, theta_std >= 0"
        super().__init__()
        self.H = H
        self.epsilon = epsilon
        self.order_2_approx = order_2_approx

        alpha_std = 10**(math.log10(abs(alpha_mean)) - 1.)
        log_dt, alpha, theta = self.init(H, dt_min, dt_max, alpha_mean, alpha_std, theta_mean, theta_std)
        self.register_parameter('log_dt', torch.nn.Parameter(log_dt))
        self.register_parameter('log_alpha', torch.nn.Parameter(alpha.log()))
        self.register_parameter('log_theta', torch.nn.Parameter(theta.log()))
        self.d = nn.Parameter(torch.ones(H))                        # [H]

    def init(self, H, dt_min, dt_max, alpha_mean, alpha_std, theta_mean, theta_std):
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
        alpha = torch.nn.init.trunc_normal_(
            torch.empty(H), mean=alpha_mean, std=alpha_std, a=self.epsilon, b=float('inf'))
        theta = torch.nn.init.trunc_normal_(
            torch.empty(H), mean=theta_mean, std=theta_std, a=self.epsilon, b=float('inf'))

        return log_dt, alpha, theta

    def forward(self, L, state=None):
        if self.order_2_approx: return self.Order2DL_forward(L)

        Delta = self.log_dt.exp().unsqueeze(0)                                   # [1 H]
        # alpha > -1
        alpha = self.log_alpha.exp().unsqueeze(0) - 1. + self.epsilon                 # [1 H]
        # theta > 0
        theta = self.log_theta.exp().unsqueeze(0) + self.epsilon                 # [1 H]
        d = self.d.unsqueeze(0)                                                  # [1 H]

        beta = 1. / theta + Delta * torch.arange(L+1, device=theta.device).unsqueeze(-1) # [L+1 H]
        k = d * theta**(-alpha-1.) / alpha * (beta[:-1,...]**(-alpha) - beta[1:,...]**(-alpha))                # [L H]

        return k
    
    def Order2DL_forward(self, L, state=None):
        raise NotImplementedError




class ExponentialExpectationKernel(nn.Module):

    def __init__(
        self,
        H,
        epsilon=1.e-6,
        dt_min=1.e-3,
        dt_max=1.e-1,
        alpha_mean=1.,
        alpha_std=0.1,
        **kwargs
    ):
        assert alpha_mean > 0 and alpha_std > 0, "alpha_mean, alpha_std, must be positive"
        super().__init__()
        self.H = H
        self.epsilon = epsilon

        alpha_std = 10**(math.log10(abs(alpha_mean)) - 1.)
        log_dt, alpha = self.init(H, dt_min, dt_max, alpha_mean, alpha_std)
        self.register_parameter('log_dt', torch.nn.Parameter(log_dt))
        self.register_parameter('log_alpha', torch.nn.Parameter(alpha.log()))
        # d (scalar coefficient) initié comme un vecteur de uns
        self.d = nn.Parameter(torch.ones(H))                        # [H]

    def init(self, H, dt_min, dt_max, alpha_mean, alpha_std):
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
        alpha = torch.nn.init.trunc_normal_(
            torch.empty(H), mean=alpha_mean, std=alpha_std, a=self.epsilon, b=float('inf'))

        return log_dt, alpha

    def forward(self, L, state=None):
        Delta = self.log_dt.exp().unsqueeze(0)                                   # [1 H]
        alpha = self.log_alpha.exp().unsqueeze(0) + self.epsilon                 # [1 H]
        d = self.d.unsqueeze(0)

        j = torch.arange(L, device=alpha.device).unsqueeze(1)
        k = d * alpha * torch.log(1 + Delta / (j * Delta + alpha))

        return k

    def Order2DL_forward(self, L, state=None):
        raise NotImplementedError




class UniformExpectationKernel(nn.Module):
    
    def __init__(
        self,
        H,
        dt_min=1e-3,
        dt_max=1e-1,
        alpha_mean=0.2,
        alpha_std=0.1,
        epsilon=1e-6,           # avoids division by 0
        order_2_approx=False,
        **kwargs
    ):
        assert alpha_mean > 0 and alpha_std >= 0, "alpha_mean should and alpha_std must be positive"
        super().__init__()
        self.H = H
        self.epsilon = epsilon
        self.order_2_approx = order_2_approx
        log_dt, alpha = self.init(H, dt_min, dt_max, alpha_mean, alpha_std)
        self.register_parameter('log_dt', torch.nn.Parameter(log_dt))
        self.register_parameter('log_alpha', torch.nn.Parameter(alpha.log()))
        self.d = nn.Parameter(torch.ones(H))

    def init(self, H, dt_min, dt_max, alpha_mean, alpha_std):
        log_dt = math.log(dt_min) + torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
        alpha = torch.nn.init.trunc_normal_(
            torch.empty(H), mean=alpha_mean, std=alpha_std, a=self.epsilon, b=float('inf'))
        return log_dt, alpha

    def forward(self, L, state=None):
        if self.order_2_approx: return self.Order2DL_forward(L)
        raise NotImplementedError

    def Order2DL_forward(self, L, state=None):
        Delta = self.log_dt.exp().unsqueeze(0)                                   # [1 H]
        # alpha is parametrized to be negative
        alpha = -self.log_alpha.exp().unsqueeze(0) - self.epsilon                 # [1 H]
        d = self.d.unsqueeze(0)                                                  # [1 H]

        j = torch.arange(L, device=alpha.device).unsqueeze(1)
        exp = Delta - Delta**2 * (j + 1./2.) * alpha/2 + Delta**3 * (j**2 + j + 1./3.) * alpha**2 / 6.
        return d * exp






class ExpectationKernel(nn.Module):
    """ Base class for other expctation kernels. #TODO
    """

    def __init__(self):
        #self.exp_X
        #self.exp_X2
        pass

    def Order2DL_forward(self):
        return
    





""" HiPPO kernel for S4 """

class HippoSSKernel(nn.Module):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=1,
        measure="legs",
        rank=1,
        channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        trainable=None, # Dictionary of options to train various HiPPO parameters
        lr=None, # Hook to set LR of hippo parameters differently
        length_correction=True, # Multiply by I-A|^L after initialization; can be turned off for initialization speed
        hurwitz=False,
        tie_state=False, # Tie parameters of HiPPO ODE across the H features
        precision=1, # 1 (single) or 2 (double) for the kernel
        resample=False  # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction
        )

    def forward(self, L=None):
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()



_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

class SSKernelNPLR(nn.Module):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    @torch.no_grad()
    def _setup_C(self, double_length=False):
        """ Construct C~ from C

        double_length: current C is for length L, convert it to length 2L
        """
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = oe.contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

    @torch.no_grad()
    def double_length(self):
        self._setup_C(double_length=True)

    def _setup_state(self):
        """ Construct dA and dB for discretized state equation """

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = torch.eye(2*self.N, dtype=C.dtype, device=C.device).unsqueeze(-2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA # (H N N)

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n') # (H N)

    def _setup_linear(self):
        """ Create parameters that allow fast linear stepping of state """
        w = self._w()
        B = _r2c(self.B) # (H N)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (torch.eye(self.rank, dtype=w.dtype, device=w.device) + 2*oe.contract('r h n, h n, s h n -> h r s', Q, D, P).real) # (H r r)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        R = torch.linalg.solve(R.to(Q_D), Q_D) # (H r N)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (r H N)
            "P": P, # (r H N)
            "Q": Q, # (r H N)
            "B": B, # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None: # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: oe.contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.size(-1) == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: oe.contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _omega(self, L, dtype, device, cache=True):
        """ Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform
        This should be called everytime the internal length self.L changes """
        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)
        if cache:
            self.register_buffer("omega", _c2r(omega))
            self.register_buffer("z", _c2r(z))
        return omega, z

    def __init__(
        self,
        L, w, P, B, C, log_dt,
        hurwitz=False,
        trainable=None,
        lr=None,
        tie_state=False,
        length_correction=True,
    ):
        """
        L: Maximum length; this module computes an SSM kernel of length L
        w: (N)
        p: (r, N) low-rank correction to A
        q: (r, N)
        A represented by diag(w) - pq^*

        B: (N)
        dt: (H) timescale per feature
        C: (H, C, N) system is 1-D to c-D (channels)

        hurwitz: tie pq and ensure w has negative real part
        trainable: toggle which of the parameters is trainable
        lr: add hook to set lr of hippo parameters specially (everything besides C)
        tie_state: tie all state parameters across the H hidden features
        length_correction: multiply C by (I - dA^L) - can be turned off when L is large for slight speedup at initialization (only relevant when N large as well)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        
        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)

        # Register parameters
        # C is a regular parameter, not state
        # self.C = nn.Parameter(_c2r(C.conj().resolve_conj()))
        #self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        self.C = nn.Parameter(_c2r(_resolve_conj(C)).clone())
        train = False
        if trainable is None: trainable = {}
        if trainable == False: trainable = {}
        if trainable == True: trainable, train = {}, True
        self.register_parameter("log_dt", torch.nn.Parameter(log_dt))
        self.register_parameter("B", torch.nn.Parameter(_c2r(B).clone()))
        self.register_parameter("P", torch.nn.Parameter(_c2r(P).clone()))
        if self.hurwitz:
            log_w_real = torch.log(-w.real + 1e-3).clone() # Some of the HiPPO methods have real part 0
            w_imag = w.imag.clone()
            self.register_parameter("log_w_real", torch.nn.Parameter(log_w_real))
            self.register_parameter("w_imag", torch.nn.Parameter(w_imag))
            self.Q = None
        else:
            self.register_parameter("w", torch.nn.Parameter(_c2r(w).clone()))
            # self.register_parameter("Q", _c2r(P.clone().conj().resolve_conj()), trainable.get('P', train), lr, 0.0)
            Q = _resolve_conj(P.clone())
            self.register_parameter("Q", torch.nn.Parameter(_c2r(Q)))

        if length_correction:
            self._setup_C()

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            w_real = -torch.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)  # (..., N)
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L > self.L:
            self.double_length()

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._w()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype, device=w.device, cache=False)

        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state # (B H N)
            sA = (
                s * _conj(w) # (B H N)
                - oe.contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]

            B = torch.cat([s, B], dim=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3) # (s+1+r, H, N)
        C = torch.cat([C, Q], dim=-3) # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L)

        # Calculate resolvent at omega
        #if has_cauchy_extension and z.dtype == torch.cfloat:
        #    r = cauchy_mult(v, z, w, symmetric=True)
        #elif has_pykeops:
        #    r = cauchy_conj(v, z, w)
        #else:
        #    r = cauchy_conj_slow(v, z, w)
        r = cauchy_conj(v, z, w)
        r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f)  # (S+1, C, H, L)

        # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L)
        else:
            k_state = None
        #k_B = k[-1, :, :, :] # (C H L)
        k_B = k.squeeze().transpose(-1, -2) # (L H)
        return k_B, k_state