import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

# ------------------- NEW: small utilities for 1D flow -------------------
class ActNorm1D(nn.Module):
    def __init__(self, D: int, dd_init: bool = True, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(D))   # b in y=(x-b)*s
        self.log_s = nn.Parameter(torch.zeros(D))  # log s
        self.dd_init = dd_init
        self.initialized = not dd_init
        self.eps = eps

    @torch.no_grad()
    def _data_init(self, x: Tensor):
        # Make y zero-mean, unit-std: y = (x - mean)/std
        mean = x.mean(dim=0)
        std  = x.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.bias.copy_(mean)
        self.log_s.copy_(torch.log(1.0 / std))
        self.initialized = True

    def forward(self, x: Tensor, reverse: bool = False):
        if self.dd_init and (not self.initialized) and (not reverse) and self.training:
            self._data_init(x)

        s = torch.exp(self.log_s).to(x.dtype)
        b = self.bias.to(x.dtype)
        if not reverse:
            y = (x - b) * s
            logdet = x.new_ones(x.size(0)) * self.log_s.sum().to(x.dtype)
        else:
            y = x * (1.0 / s) + b
            logdet = x.new_ones(x.size(0)) * (-self.log_s.sum().to(x.dtype))
        return y, logdet

class OrthogonalMixing1D(nn.Module):
    """Householder sequence mixing: orthogonal W with log|det W| = 0."""
    def __init__(self, D: int, n_reflections: int = 4):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n_reflections, D) * 0.02)

    @staticmethod
    def _householder(x: Tensor, v: Tensor) -> Tensor:
        # x: (B,D), v: (D,) normalized
        v = F.normalize(v, dim=-1)
        proj = (x * v).sum(dim=-1, keepdim=True)       # (B,1)
        return x - 2.0 * proj * v.unsqueeze(0)         # reflect

    def forward(self, x: Tensor, reverse: bool = False):
        # Orthogonal => inverse == transpose == apply reflections in reverse order
        if not reverse:
            y = x
            for i in range(self.v.size(0)):
                y = self._householder(y, self.v[i])
        else:
            y = x
            for i in reversed(range(self.v.size(0))):
                y = self._householder(y, self.v[i])
        logdet = x.new_zeros(x.size(0))
        return y, logdet

# ------------------- Upgraded conditional affine coupling (1D) -------------------
# ------------------- NEW: small utilities for 1D flow -------------------
class ActNorm1D(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(D))
        self.log_s = nn.Parameter(torch.zeros(D))

    def forward(self, x: Tensor, reverse: bool = False):
        s = torch.exp(self.log_s).to(x.dtype)  # (D,)
        b = self.bias.to(x.dtype)
        if not reverse:
            y = (x - b) * s
            # old (buggy): logdet = x.new_full((x.size(0),), self.log_s.sum().to(x.dtype))
            logdet = x.new_ones(x.size(0)) * self.log_s.sum().to(x.dtype)
        else:
            y = x * (1.0 / s) + b
            # old (buggy): logdet = x.new_full((x.size(0),), -self.log_s.sum().to(x.dtype))
            logdet = -x.new_ones(x.size(0)) * self.log_s.sum().to(x.dtype)
        return y, logdet


class OrthogonalMixing1D(nn.Module):
    """Householder sequence mixing: orthogonal W with log|det W| = 0."""
    def __init__(self, D: int, n_reflections: int = 4):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n_reflections, D) * 0.02)

    @staticmethod
    def _householder(x: Tensor, v: Tensor) -> Tensor:
        # x: (B, D), v: (D,) normalized
        v = F.normalize(v, dim=-1)
        proj = (x * v).sum(dim=-1, keepdim=True)  # (B, 1)
        return x - 2.0 * proj * v.unsqueeze(0)    # reflect

    def forward(self, x: Tensor, reverse: bool = False):
        # Orthogonal => inverse == transpose == apply reflections in reverse order
        if not reverse:
            y = x
            for i in range(self.v.size(0)):
                y = self._householder(y, self.v[i])
        else:
            y = x
            for i in reversed(range(self.v.size(0))):
                y = self._householder(y, self.v[i])
        logdet = x.new_zeros(x.size(0))
        return y, logdet


# ------------------- Upgraded conditional affine coupling (1D) -------------------
class _CondAffineCoupling1D(nn.Module):
    """
    RealNVP-style 1D affine coupling with a deeper context-gated ResNet and even/odd masking.
    """
    def __init__(
        self,
        D: int,
        context_dim: int,
        mask_even: bool,
        hidden: int = 512,
        n_blocks: int = 3,
        scale_clamp: float = 5.0,
    ):
        super().__init__()
        self.D = D
        self.scale_clamp = float(scale_clamp)

        # even/odd mask (like before)
        mask = torch.zeros(D)
        mask[::2] = 1.0
        if not mask_even:
            mask = 1.0 - mask
        self.register_buffer("mask", mask)  # (D,)

        in_dim = int(self.mask.sum().item())  # identity pass subset
        out_dim = D - in_dim                  # transformed subset

        # Input projection for x_sel and a context stem
        self.x_proj = nn.Linear(in_dim, hidden)
        self.c_stem = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Context-gated residual blocks
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.ModuleDict({
                "ln": nn.LayerNorm(hidden),
                "fc": nn.Linear(hidden, hidden),
                "c_gate": nn.Linear(hidden, 2 * hidden),  # FiLM: gamma, beta
            }))
        self.blocks = nn.ModuleList(blocks)

        # Output head to (s,t); zero-init to start near identity
        self.out = nn.Linear(hidden, 2 * out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, c: Tensor, reverse: bool = False):
        B, D = x.shape
        m = self.mask.bool()               # (D,)
        x_sel = x[:, m]                    # (B, in_dim)

        # Build representation with FiLM gating from context
        h = self.x_proj(x_sel).to(x.dtype)  # (B, H)
        c_feat = self.c_stem(c).to(x.dtype) # (B, H)

        for blk in self.blocks:
            h_in = h
            h = blk["ln"](h)
            h = F.silu(h)
            # FiLM from context
            gamma, beta = blk["c_gate"](c_feat).chunk(2, dim=-1)
            h = h * (1.0 + torch.tanh(gamma)) + beta
            h = h + blk["fc"](h)           # residual

        s, t = self.out(h).chunk(2, dim=-1)  # (B, out_dim) each
        # gentle scaling for stability
        s = torch.tanh(s) * self.scale_clamp
        s = s.to(x.dtype)
        t = t.to(x.dtype)

        y = x.clone()
        idx = (~m)
        if not reverse:
            exp_s = torch.exp(s.float()).to(x.dtype)
            y[:, idx] = x[:, idx] * exp_s + t
            logdet = s.sum(dim=-1)
        else:
            exp_ns = torch.exp((-s).float()).to(x.dtype)
            y[:, idx] = (x[:, idx] - t) * exp_ns
            logdet = -s.sum(dim=-1)

        return y, logdet

def diag_gauss_sample_and_logq(m, logstd):
    z = torch.randn_like(m)
    eps = m + torch.exp(logstd) * z
    D = m.shape[-1]
    logq = -0.5 * (((eps - m) * torch.exp(-logstd)).pow(2).sum(-1)
                   + 2.0 * logstd.sum(-1) + D * math.log(2.0 * math.pi))
    return eps, logq  # shapes: (B,D), (B,)

def logp_delta_under_Q(delta, L_Q, logdetQ):
    # log N(delta; 0, Q) with Q = L_Q L_Q^T
    orig = delta.shape
    x = delta.reshape(-1, orig[-1]).unsqueeze(-1)                 # (N,D,1)
    sol = torch.cholesky_solve(x, L_Q).squeeze(-1)               # (N,D)
    quad = (x.squeeze(-1) * sol).sum(-1)                         # (N,)
    D = orig[-1]
    logp = -0.5 * (quad + logdetQ + D * math.log(2.0 * math.pi))
    return logp.view(*orig[:-1])                                 # (...,)

class GaussianShockHead(nn.Module):
    def __init__(self, D, context_dim, hidden=512, mean_bound=0.5, fixed_logstd=-3.0):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, D)
        )
        self.mean_bound = float(mean_bound)
        self.register_buffer("fixed_logstd", torch.tensor(float(fixed_logstd)))

    def forward(self, c):
        m = self.mean_bound * torch.tanh(self.mu(c))          # bound mean to [-B, B]
        ls = self.fixed_logstd.expand_as(m)                   # constant log-std
        return m, ls

# --- NEW: per‑pixel Gaussian shock head for 2‑D states -----------------
class GaussianShockHead2D(nn.Module):
    """
    Input:  c_map (B, Cin, H, W) where Cin = 1 (state) + ctx_ch (per-step) + 1 (time)
    Output: (m, logstd) both of shape (B, 1, H, W)  (after padding, if needed)
    """
    def __init__(self, in_ch: int, hidden: int = 64, mean_bound: float = 0.5, fixed_logstd: float = -3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1, padding_mode="circular"),
        )
        # start close to identity: zero last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.mean_bound = float(mean_bound)
        self.register_buffer("fixed_logstd", torch.tensor(float(fixed_logstd)))

    def forward(self, c_map: Tensor):
        # Normalize to (B, C, H, W)
        if c_map.dim() == 3:            # (B, H, W) -> (B, 1, H, W)
            c_map = c_map.unsqueeze(1)
        elif c_map.dim() == 2:          # (H, W) -> (1, 1, H, W)
            c_map = c_map.unsqueeze(0).unsqueeze(0)

        B, C, H, W = c_map.shape

        # If "top-half" (H is exactly half of width), pad zeros to the bottom.
        if (2 * H == W) and (W % 2 == 0):
            full = c_map.new_zeros(B, C, 2 * H, W)
            full[:, :, :H, :] = c_map
            c_map = full
        # (Optional) If ever "left-half" (W is half of height), pad zeros to the right.
        elif (2 * W == H) and (H % 2 == 0):
            full = c_map.new_zeros(B, C, H, 2 * W)
            full[:, :, :, :W] = c_map
            c_map = full

        m = self.mean_bound * torch.tanh(self.net(c_map))
        ls = self.fixed_logstd.to(dtype=m.dtype, device=m.device).expand_as(m)
        return m, ls
'''
class GaussianShockHead2D(nn.Module):
    """
    Input:  c_map (B, Cin, H, W) where Cin = 1 (state) + ctx_ch (per‑step) + 1 (time)
    Output: (m, logstd) both of shape (B, 1, H, W)
    """
    def __init__(self, in_ch: int, hidden: int = 64, mean_bound: float = 0.5, fixed_logstd: float = -3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1, padding_mode="circular"),
        )
        # start close to identity: zero last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.mean_bound = float(mean_bound)
        self.register_buffer("fixed_logstd", torch.tensor(float(fixed_logstd)))

    def forward(self, c_map: Tensor):
        m = self.mean_bound * torch.tanh(self.net(c_map))
        ls = self.fixed_logstd.expand_as(m)
        return m, ls
''' 
class SinPosEnc(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (S,B,E)
        S = x.size(0)
        return x + self.pe[:S].unsqueeze(1)

def causal_mask(sz, device):
    m = torch.full((sz, sz), float("-inf"), device=device)
    return torch.triu(m, 1)

class TransformerFutureEncoder1D(nn.Module):
    """
    Inputs:
      Y: (B, T, D)  e.g., (batch, steps, L)
    Outputs:
      c_global: (B, Cg)
      c_steps : (B, T-1, Cs)   where each c_steps[:,t] sees Y_{t:T} (future-aware)
    """
    def __init__(self, d_in: int, d_model=512, nhead=8, layers=8, ff=512, Cg=256, Cs=128, dropout=0.0):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=ff, dropout=dropout,
                                               batch_first=False, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pos = SinPosEnc(d_model)
        self.proj_global = nn.Linear(d_model, Cg)
        self.proj_steps  = nn.Linear(d_model, Cs)

    def forward(self, Y):  # (B,T,D)
        B, T, D = Y.shape
        Y_rev = torch.flip(Y, dims=[1])                 # reverse time → future becomes "past"
        x = self.proj_in(Y_rev).transpose(0, 1)         # (T,B,d_model)
        x = self.pos(x)
        mask = causal_mask(T, Y.device)                 # causal over reversed time
        h = self.encoder(x, mask=mask).transpose(0, 1)  # (B,T,d_model)
        h = torch.flip(h, dims=[1])                     # back to original order (B,T,d_model)

        # global summary (mean pool) and per-step features except last step
        c_global = self.proj_global(h.mean(dim=1))      # (B,Cg)
        #c_steps  = self.proj_steps(h[:, :-1, :])        # (B,T-1,Cs) → c_t uses Y_{t:T}
        c_steps  = self.proj_steps(h[:, 1:, :])        # (B,T-1,Cs) → c_t uses Y_{t+1:T}
        return c_global, c_steps

def _pad_top_half_to_full_flat(x: torch.Tensor, N: int) -> torch.Tensor:
    """
    x: (B, H, W) or (B, T, H, W) with either H×W = N×N (full) or (N/2)×N (top half).
    returns flattened (B, D) or (B, T, D) with D=N*N, zero-filling the bottom half if needed.
    """
    if x.ndim == 4:  # (B,T,H,W)
        B, T, H, W = x.shape
        if (H, W) == (N, N):
            return x.reshape(B, T, N*N)
        if (H, W) == (N//2, N):
            full = x.new_zeros(B, T, N, N)
            full[:, :, :N//2, :] = x
            return full.reshape(B, T, N*N)
        # fallback: resize to full
        full = F.interpolate(x.reshape(B*T, 1, H, W), size=(N, N),
                             mode="bilinear", align_corners=False).reshape(B, T, N, N)
        return full.reshape(B, T, N*N)
    else:
        raise ValueError(f"unexpected shape for shock context: {tuple(x.shape)}")
    
class NoisyGaussianB1(nn.Module):
    """
    States-out, shocks-inside (Gaussian). Uses:
      - flow_x1: your cNF for q(X1 | Y_{1:T})
      - shock_head: GaussianShockHead for per-step eps
      - f_theta: physics step; Q: process noise (D,D)
    API mirrors your flow: sample_and_logq returns flattened X and log q.
    """
    def __init__(self, D, steps, f_theta, flow_x1, shock_head, Q):
        super().__init__()
        self.D, self.T = D, steps
        self.f_theta = f_theta
        self.flow_x1 = flow_x1
        self.shock_head = shock_head
        L_Q = torch.linalg.cholesky(Q)
        self.register_buffer("L_Q", L_Q)
        self.register_buffer("logdetQ", 2.0 * torch.log(torch.diag(L_Q)).sum())

    def _timefrac(self, B, t, device, dtype):
        return torch.full((B,1), float(t)/max(self.T-1,1), device=device, dtype=dtype)

    def sample_rollout_and_logq(self, B, c_global, c_steps, device):
        # X1 ~ q(X1 | Y)
        x1, logq_x1 = self.flow_x1.sample_and_logq(B=B, D=self.D, c=c_global, device=device)
        Xs = [x1]; logq_eps_sum = torch.zeros(B, device=device, dtype=x1.dtype)
        deltas = []; logp_dyn_sum = torch.zeros(B, device=device, dtype=x1.dtype)

        for t in range(self.T-1):
            X_t = Xs[-1]
            time_frac = self._timefrac(B, t, device, X_t.dtype)
            c_t = torch.cat([X_t, c_steps[:, t, :], time_frac], dim=-1)   # (B, D+Cs+1)
            #c_t = torch.cat([c_steps[:, t, :], time_frac], dim=-1)   # (B, Cs+1)
            m_t, logstd_t = self.shock_head(c_t)                          # (B,D),(B,D)
            delta_t, logq_eps_t = diag_gauss_sample_and_logq(m_t, logstd_t)
            X_next = torch.clamp(self.f_theta(X_t) + delta_t, -20, 20)
            assert torch.isfinite(X_next).all(), [X_next[0], delta_t[0]]
            Xs.append(X_next)

            logq_eps_sum += logq_eps_t
            deltas.append(delta_t)
            logp_dyn_sum += logp_delta_under_Q(delta_t, self.L_Q, self.logdetQ)

        X = torch.stack(Xs, dim=1)                # (B,T,D)
        delta_seq = torch.stack(deltas, dim=1)    # (B,T-1,D)
        x_flat = X.reshape(B, -1)
        logq_total = logq_x1 + logq_eps_sum
        aux = {"X_seq": X, "delta_seq": delta_seq, "logp_dyn_sum": logp_dyn_sum,
               "logq_x1": logq_x1, "logq_eps_sum": logq_eps_sum}
        return x_flat, logq_total, aux

    # convenience to match your trainer's (B,D,c,device) signature; c=(c_global,c_steps) or a single tensor
    def sample_and_logq(self, B, D, c, device):
        if isinstance(c, (tuple, list)) and len(c)==2:
            c_global, c_steps = c
        else:
            c_global, c_steps = c, c.unsqueeze(1).repeat(1, self.T-1, 1)
        x_flat, logq, _ = self.sample_rollout_and_logq(B, c_global, c_steps, device)
        return x_flat, logq
    
# ------------------- Stronger 1D flow (stacks coupling + mixing + actnorm) -------------------
class RealNVP_Flow1D(nn.Module):
    """
    Stack [ActNorm -> Coupling -> OrthogonalMixing] x n_layers.
    Orthogonal mixing breaks alignment of the binary mask without extra logdet cost.
    """
    def __init__(
        self,
        D: int,
        context_dim: int,
        n_layers: int = 12,
        hidden: int = 512,
        n_reflections: int = 4,
        scale_clamp: float = 5.0,
        use_actnorm: bool = True,
    ):
        super().__init__()
        self.D = D
        layers = []
        for i in range(n_layers):
            if use_actnorm:
                layers.append(ActNorm1D(D))
            layers.append(_CondAffineCoupling1D(
                D, context_dim, mask_even=(i % 2 == 0),
                hidden=hidden, n_blocks=3, scale_clamp=scale_clamp
            ))
            layers.append(OrthogonalMixing1D(D, n_reflections=n_reflections))
        self.layers = nn.ModuleList(layers)

    def forward(self, z: Tensor, c: Tensor):
        logdet = z.new_zeros(z.size(0))
        x = z
        for layer in self.layers:
            if isinstance(layer, _CondAffineCoupling1D):
                x, ld = layer(x, c, reverse=False)
            else:
                x, ld = layer(x, reverse=False)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: Tensor, c: Tensor):
        logdet = x.new_zeros(x.size(0))
        z = x
        for layer in reversed(self.layers):
            if isinstance(layer, _CondAffineCoupling1D):
                z, ld = layer(z, c, reverse=True)
            else:
                z, ld = layer(z, reverse=True)
            logdet = logdet + ld
        return z, logdet

    def sample_and_logq(self, B: int, D: int, c: Tensor, device):
        z = torch.randn(B, D, device=device, dtype=c.dtype)
        x, logdet = self.forward(z, c)
        logpz = -0.5 * (z ** 2).sum(-1) - 0.5 * D * math.log(2.0 * math.pi)
        logq = logpz - logdet
        return x, logq

# ------------------- rational-quadratic spline core -------------------

@torch.no_grad()
def _softplus_inv(y: float) -> float:
    # inverse softplus for scalar y>0
    return float(math.log(math.expm1(y)))

def _calc_knots(lengths: Tensor, lo: float, hi: float):
    """lengths in (B, D, K) that sum to 1; returns (lengths_scaled, knots) with knots in [lo,hi]."""
    K = lengths.size(-1)
    # cumulative in [0,1]
    knots01 = torch.cumsum(lengths, dim=-1)
    knots01 = F.pad(knots01, (1, 0), value=0.0)
    # scale/shift
    knots = (hi - lo) * knots01 + lo
    # make sure exact endpoints and positive lengths after fp accumulations
    knots[..., 0] = lo
    knots[..., -1] = hi
    scaled = knots[..., 1:] - knots[..., :-1]
    return scaled, knots

def _select_gather(x: Tensor, idx: Tensor) -> Tensor:
    # x: (B, D, K or K+1), idx: (B, D)
    idx = idx.clamp(min=0, max=x.size(-1) - 1)
    return x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

def rational_quadratic_spline(
    inputs: Tensor,
    widths_logits: Tensor,   # (B, D, K)
    heights_logits: Tensor,  # (B, D, K)
    deriv_logits: Tensor,    # (B, D, K-1)  -- interior knot derivatives
    *,
    inverse: bool = False,
    bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
    eps: float = 1e-6,
):
    """
    Vectorized RQ-spline with linear tails outside [-bound, bound].
    Computes in float32 for numerical stability, then casts back to inputs.dtype.
    Returns (outputs, logabsdet) in the same dtype as `inputs`.
    """
    orig_dtype = inputs.dtype
    calc_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    device = inputs.device

    # Cast everything to a common compute dtype
    x = inputs.to(calc_dtype)
    wl = widths_logits.to(calc_dtype)
    hl = heights_logits.to(calc_dtype)
    dl = deriv_logits.to(calc_dtype)

    Bdim, Ddim = x.shape
    K = wl.size(-1)
    if min_bin_width * K > 1.0:
        raise ValueError("min_bin_width too large for K.")
    if min_bin_height * K > 1.0:
        raise ValueError("min_bin_height too large for K.")

    # ---- helpers (compute-dtype) ----
    def _calc_knots(lengths: Tensor, lo: float, hi: float):
        K = lengths.size(-1)
        knots01 = torch.cumsum(lengths, dim=-1)
        knots01 = F.pad(knots01, (1, 0), value=0.0)
        knots = (hi - lo) * knots01 + lo
        knots[..., 0] = lo
        knots[..., -1] = hi
        scaled = knots[..., 1:] - knots[..., :-1]
        return scaled, knots

    def _select_gather(t: Tensor, idx: Tensor) -> Tensor:
        idx = idx.clamp(min=0, max=t.size(-1) - 1)
        return t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    def _softplus_inv_scalar(y: float) -> float:
        return float(math.log(math.expm1(y)))

    # Normalize to simplex + enforce minimums
    widths = F.softmax(wl, dim=-1)
    heights = F.softmax(hl, dim=-1)
    widths  = min_bin_width  + (1.0 - K * min_bin_width)  * widths
    heights = min_bin_height + (1.0 - K * min_bin_height) * heights

    # Derivatives: bias so zero logits -> ~1.0 slope
    deriv_bias = _softplus_inv_scalar(1.0 - min_derivative)
    deriv = F.softplus(dl + deriv_bias) + min_derivative       # (B,D,K-1)
    deriv = F.pad(deriv, (1, 1), value=1.0 - min_derivative)   # (B,D,K+1)

    left = -bound
    right = bound
    bottom = -bound
    top = bound

    inside = (x >= left) & (x <= right)

    # Preallocate (compute-dtype); start with identity (good for tails)
    outputs = x.clone()
    logabsdet = torch.zeros_like(x)

    # Knot positions and sizes
    widths_x, cumx = _calc_knots(widths, left, right)     # (B,D,K), (B,D,K+1)
    heights_y, cumy = _calc_knots(heights, bottom, top)   # (B,D,K), (B,D,K+1)

    edges = cumy if inverse else cumx
    bin_idx = torch.searchsorted(edges.contiguous(), x.unsqueeze(-1), right=False).squeeze(-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)  # (B,D)

    # Gather per element params
    w  = _select_gather(widths_x, bin_idx)                 # (B,D)
    x0 = _select_gather(cumx,     bin_idx)
    x1 = x0 + w

    h  = _select_gather(heights_y, bin_idx)
    y0 = _select_gather(cumy,      bin_idx)
    y1 = y0 + h

    d0 = _select_gather(deriv,            bin_idx)
    d1 = _select_gather(deriv[..., 1:],   bin_idx)
    delta = h / w

    if inverse:
        y = x
        a = (y - y0) * (d0 + d1 - 2.0 * delta) + h * (delta - d0)
        b = h * d0 - (y - y0) * (d0 + d1 - 2.0 * delta)
        c = -delta * (y - y0)
        disc = (b * b - 4.0 * a * c).clamp_min(0.0)
        # Stable root
        t = (2.0 * c) / (-b - torch.sqrt(disc) + eps)
        t = t.clamp(0.0, 1.0)

        x_new = x0 + t * w
        outputs[inside] = x_new[inside]

        tmt = t * (1.0 - t)
        denom = delta + (d0 + d1 - 2.0 * delta) * tmt
        num = (delta ** 2) * (d1 * t.pow(2) + 2.0 * delta * tmt + d0 * (1.0 - t).pow(2))
        lad = -(torch.log(num + eps) - 2.0 * torch.log(denom + eps))
        logabsdet[inside] = lad[inside]
    else:
        t = ((x - x0) / w).clamp(0.0, 1.0)
        tmt = t * (1.0 - t)
        num = h * (delta * t.pow(2) + d0 * tmt)
        denom = delta + (d0 + d1 - 2.0 * delta) * tmt
        y = y0 + num / (denom + eps)
        outputs[inside] = y[inside]

        num_d = (delta ** 2) * (d1 * t.pow(2) + 2.0 * delta * tmt + d0 * (1.0 - t).pow(2))
        lad = torch.log(num_d + eps) - 2.0 * torch.log(denom + eps)
        logabsdet[inside] = lad[inside]

    # Cast back to the original dtype to match the caller (prevents bf16/fp32 mismatch)
    return outputs.to(orig_dtype), logabsdet.to(orig_dtype)

# ------------------- conditional spline coupling (1D) -------------------

class _CondSplineCoupling1D(nn.Module):
    """
    RealNVP-style 1D coupling where the transformed subset uses a
    conditional monotonic RQ-spline instead of affine.

    Params per transformed dim = 3*K - 1 (K widths, K heights, K-1 interior derivatives).
    """
    def __init__(
        self,
        D: int,
        context_dim: int,
        mask_even: bool,
        hidden: int = 512,
        n_blocks: int = 3,
        num_bins: int = 8,
        bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.D = D
        self.num_bins = int(num_bins)
        self.bound = float(bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        # even/odd mask
        mask = torch.zeros(D)
        mask[::2] = 1.0
        if not mask_even:
            mask = 1.0 - mask
        self.register_buffer("mask", mask)

        in_dim  = int(self.mask.sum().item())   # identity pass subset
        out_dim = D - in_dim                    # transformed subset
        self.out_dim = out_dim

        # Input projection + context stem (FiLM-gated residual trunk)
        self.x_proj = nn.Linear(in_dim, hidden)
        self.c_stem = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.ModuleDict({
                "ln": nn.LayerNorm(hidden),
                "fc": nn.Linear(hidden, hidden),
                "c_gate": nn.Linear(hidden, 2 * hidden),   # FiLM: gamma, beta
            }))
        self.blocks = nn.ModuleList(blocks)

        # Head: raw logits for widths/heights + logits for interior derivatives
        head_dim = out_dim * (2 * self.num_bins + (self.num_bins - 1))
        self.out = nn.Linear(hidden, head_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # near-uniform bins; derivatives ~1 due to bias in spline fn

    def _predict_params(self, h: Tensor):
        """Split and reshape head outputs to (B, out_dim, K/K-1)."""
        B = h.size(0)
        K = self.num_bins
        H = self.out(h)
        s1 = self.out_dim * K
        s2 = s1 + self.out_dim * K
        widths_logits  = H[:, :s1].view(B, self.out_dim, K)
        heights_logits = H[:, s1:s2].view(B, self.out_dim, K)
        deriv_logits   = H[:, s2:].view(B, self.out_dim, K - 1)  # interior only
        return widths_logits, heights_logits, deriv_logits

    def forward(self, x: Tensor, c: Tensor, reverse: bool = False):
        B, D = x.shape
        m = self.mask.bool()
        x_id = x[:, m]       # pass-through dims
        x_tr = x[:, ~m]      # transformed dims

        # Build features (FiLM)
        h = self.x_proj(x_id).to(x.dtype)
        c_feat = self.c_stem(c).to(x.dtype)
        for blk in self.blocks:
            h_in = h
            h = blk["ln"](h)
            h = F.silu(h)
            gamma, beta = blk["c_gate"](c_feat).chunk(2, dim=-1)
            h = h * (1.0 + torch.tanh(gamma)) + beta
            h = h + blk["fc"](h)  # residual

        wl, hl, dl = self._predict_params(h)  # (B, out_dim, K/K-1)

        # Apply spline elementwise on transformed subset
        if not reverse:
            y_tr, lad = rational_quadratic_spline(
                x_tr, wl, hl, dl,
                inverse=False,
                bound=self.bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            logdet = lad.sum(dim=-1)
        else:
            z_tr, lad = rational_quadratic_spline(
                x_tr, wl, hl, dl,
                inverse=True,
                bound=self.bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            y_tr = z_tr
            logdet = -lad.sum(dim=-1)

        # Merge back
        y = x.clone()
        y[:, m]  = x_id
        y[:, ~m] = y_tr
        return y, logdet


# ------------------- stronger 1D flow (spline coupling) -------------------

class Spline_Flow1D(nn.Module):
    """
    Stack [ActNorm -> SplineCoupling -> OrthogonalMixing] x n_layers.
    Orthogonal mixing breaks alignment of the binary mask with zero logdet.
    """
    def __init__(
        self,
        D: int,
        context_dim: int,
        n_layers: int = 12,
        hidden: int = 512,
        n_reflections: int = 4,
        num_bins: int = 8,
        bound: float = 3.0,
        use_actnorm: bool = True,
        # legacy arg kept for API compatibility; not used by spline
        scale_clamp: float = 5.0,
    ):
        super().__init__()
        self.D = D
        layers = []
        for i in range(n_layers):
            if use_actnorm:
                layers.append(ActNorm1D(D))
            layers.append(_CondSplineCoupling1D(
                D=D, context_dim=context_dim, mask_even=(i % 2 == 0),
                hidden=hidden, n_blocks=3,
                num_bins=num_bins, bound=bound,
            ))
            layers.append(OrthogonalMixing1D(D, n_reflections=n_reflections))
        self.layers = nn.ModuleList(layers)

    def forward(self, z: Tensor, c: Tensor):
        logdet = z.new_zeros(z.size(0))
        x = z
        for layer in self.layers:
            if isinstance(layer, _CondSplineCoupling1D):
                x, ld = layer(x, c, reverse=False)
            else:
                x, ld = layer(x, reverse=False)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, x: Tensor, c: Tensor):
        logdet = x.new_zeros(x.size(0))
        z = x
        for layer in reversed(self.layers):
            if isinstance(layer, _CondSplineCoupling1D):
                z, ld = layer(z, c, reverse=True)
            else:
                z, ld = layer(z, reverse=True)
            logdet = logdet + ld
        return z, logdet

    def sample_and_logq(self, B: int, D: int, c: Tensor, device):
        z = torch.randn(B, D, device=device, dtype=c.dtype)
        x, logdet = self.forward(z, c)
        logpz = -0.5 * (z ** 2).sum(-1) - 0.5 * D * math.log(2.0 * math.pi)
        logq = logpz - logdet
        return x, logq    

def squeeze2x(x):
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    x = x.view(B, C, H//2, 2, W//2, 2).permute(0,1,3,5,2,4).contiguous()
    return x.view(B, C*4, H//2, W//2)

def unsqueeze2x(x):
    B, C, H, W = x.shape
    assert C % 4 == 0
    x = x.view(B, C//4, 2, 2, H, W).permute(0,1,4,2,5,3).contiguous()
    return x.view(B, C//4, H*2, W*2)

# ---------- Glow-style invertible 1×1 conv (channel mixer) ----------
class Inv1x1Conv(nn.Module):
    def __init__(self, C: int, identity_init: bool = True):
        super().__init__()
        W = torch.eye(C) if identity_init else torch.linalg.qr(torch.randn(C, C))[0]
        self.weight = nn.Parameter(W.view(C, C, 1, 1))

    def forward(self, x: Tensor, reverse: bool = False):
        B, C, H, W = x.shape
        Wmat = self.weight.view(C, C)
        # log|det(W)| * (#spatial sites) per sample
        slogdet = torch.slogdet(Wmat).logabsdet * (H * W)
        if not reverse:
            y = F.conv2d(x, self.weight)
            return y, slogdet.expand(B)
        else:
            Winv = torch.inverse(Wmat).view(C, C, 1, 1)
            y = F.conv2d(x, Winv)
            return y, (-slogdet).expand(B)

# ------------------- 2D conditional affine coupling -------------------
class _AffineCoupling2D(nn.Module):
    """2D RealNVP coupling with circular padding and context.
       checkerboard=True  -> spatial checkerboard mask
       checkerboard=False -> CHANNEL-wise half/half mask (for squeezed tensors)
       channel_flip toggles which half is identity.
    """
    def __init__(self,
                 in_ch: int = 1,
                 ctx_ch: int = 64,
                 hidden: int = 64,
                 checkerboard: bool = True,
                 channel_flip: bool = False,
                 scale_clamp: float = 1.0):
        super().__init__()
        self.checkerboard = checkerboard
        self.channel_flip = channel_flip
        self.scale_clamp = scale_clamp
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + ctx_ch, hidden, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"), nn.SiLU(),
            nn.Conv2d(hidden, 2 * in_ch, 3, padding=1, padding_mode="circular"),
        )
        # identity-init the last conv so the whole layer starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self._mask_cache_checker = None  # (1,1,H,W)

    def _mask(self, x: Tensor):
        B, C, H, W = x.shape
        if self.checkerboard:
            if (self._mask_cache_checker is None) or (self._mask_cache_checker.shape[-2:] != (H, W)):
                i = torch.arange(H, device=x.device)[:, None]
                j = torch.arange(W, device=x.device)[None, :]
                cb = ((i + j) % 2).float()[None, None, :, :]
                self._mask_cache_checker = cb
            return self._mask_cache_checker
        else:
            m = x.new_zeros(1, C, 1, 1)
            if not self.channel_flip:
                m[:, :C // 2] = 1.0      # first half identity
            else:
                m[:, C // 2:] = 1.0      # second half identity
            return m

    def forward(self, x: Tensor, c_map: Tensor, reverse: bool = False):
        # x: (B,C,H,W), c_map: (B,Cc,H,W)
        m = self._mask(x)                    # (1,1,H,W) or (1,C,1,1)
        xm = x * m
        # --- make c_map match xm spatially (handles top-half & pyramid levels) ---
        c_use = c_map
        # if a pyramid/list is passed per level, pick the current one
        if isinstance(c_use, (list, tuple)):
            # L is the current level index in your surrounding loop; if not available, pick 0
            # or pass the per-level context when calling this block.
            c_use = c_use[L] if 'L' in locals() else c_use[0]
        # ensure (B,C,H,W)
        if c_use.dim() == 3:
            c_use = c_use.unsqueeze(1)
        # if it's a top-half (H, W) = (N/2, N), pad to (N, N)
        Bc, Cc, Hc, Wc = c_use.shape
        if (2 * Hc == Wc) and (Wc % 2 == 0):
            full = c_use.new_zeros(Bc, Cc, 2 * Hc, Wc)
            full[:, :, :Hc, :] = c_use
            c_use = full
        # match xm's spatial size
        Ht, Wt = xm.shape[-2], xm.shape[-1]
        Hc, Wc = c_use.shape[-2], c_use.shape[-1]
        if (Hc, Wc) != (Ht, Wt):
            if (Hc % Ht == 0) and (Wc % Wt == 0):
                kh, kw = Hc // Ht, Wc // Wt
                if kh > 1 or kw > 1:
                    c_use = F.avg_pool2d(c_use, kernel_size=(kh, kw), stride=(kh, kw))
            if c_use.shape[-2:] != (Ht, Wt):  # still off → interpolate
                c_use = F.interpolate(c_use, size=(Ht, Wt), mode="bilinear", align_corners=False)
        # keep dtype/device aligned
        c_use = c_use.to(dtype=xm.dtype, device=xm.device, non_blocking=True)
        # concatenate along channels
        h = torch.cat([xm, c_use], dim=1)

        st = self.net(h)
        s, t = torch.chunk(st, 2, dim=1)
        s = torch.tanh(s) * self.scale_clamp  # gentler early scaling
        if not reverse:
            y = xm + (1 - m) * (x * torch.exp(s) + t)
            logdet = ((1 - m) * s).sum(dim=[1, 2, 3])
        else:
            y = xm + (1 - m) * ((x - t) * torch.exp(-s))
            logdet = -((1 - m) * s).sum(dim=[1, 2, 3])
        return y, logdet

# ------------------- 2D flow with squeeze + channel masks + 1x1 conv -------------------
class _Flow2DLevel(nn.Module):
    def __init__(self, C: int, Cctx: int, n_blocks: int, use_1x1: bool = True, scale_clamp: float = 1.0):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(_AffineCoupling2D(in_ch=C, ctx_ch=Cctx,
                                            checkerboard=False,
                                            channel_flip=(i % 2 == 1),
                                            scale_clamp=scale_clamp))
            if use_1x1:
                blocks.append(Inv1x1Conv(C, identity_init=True))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, c_map: Tensor, reverse: bool = False):
        logdet = x.new_zeros(x.size(0))
        if not reverse:
            for b in self.blocks:
                if isinstance(b, Inv1x1Conv):
                    x, sld = b(x, reverse=False); logdet = logdet + sld
                else:
                    x, ld  = b(x, c_map, reverse=False); logdet = logdet + ld
        else:
            for b in reversed(self.blocks):
                if isinstance(b, Inv1x1Conv):
                    x, sld = b(x, reverse=True);  logdet = logdet + sld
                else:
                    x, ld  = b(x, c_map, reverse=True); logdet = logdet + ld
        return x, logdet

class _Flow2D(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 ctx_ch: int = 64,
                 n_layers: int = 8,           # kept for backward compat (single-level)
                 use_squeeze: bool = True,
                 use_1x1: bool = True,
                 scale_clamp: float = 1.0,
                 squeeze_levels: int = 2,     # <<< NEW: number of squeeze stages
                 blocks_per_level: int | tuple = (4, 4)):  # blocks per level
        super().__init__()
        self.use_squeeze = use_squeeze
        self.squeeze_levels = squeeze_levels if use_squeeze else 0
        if isinstance(blocks_per_level, int):
            blocks_per_level = [blocks_per_level] * max(self.squeeze_levels, 1)

        # Build a level after each squeeze
        levels = []
        C, Cctx = in_ch, ctx_ch
        for L in range(self.squeeze_levels or 1):
            if self.use_squeeze:
                C    *= 4     # space-to-depth
                Cctx *= 4
            n_blocks = blocks_per_level[L] if self.squeeze_levels else n_layers
            levels.append(_Flow2DLevel(C, Cctx, n_blocks=n_blocks,
                                       use_1x1=use_1x1, scale_clamp=scale_clamp))
        self.levels = nn.ModuleList(levels)

    def forward(self, z: Tensor, c_map: Tensor):
        x, c = z, c_map
        logdet = z.new_zeros(z.size(0))
        # go down (squeeze → blocks)
        for L in range(self.squeeze_levels or 1):
            if self.use_squeeze:
                x = squeeze2x(x); c = squeeze2x(c)
            x, ld = self.levels[L](x, c, reverse=False)
            logdet = logdet + ld
        # go up (unsqueeze)
        for _ in range(self.squeeze_levels):
            x = unsqueeze2x(x)
        return x, logdet

    def inverse(self, x: Tensor, c_map: Tensor):
        z, c = x, c_map
        logdet = x.new_zeros(x.size(0))
        # go down
        for L in range(self.squeeze_levels or 1):
            if self.use_squeeze:
                z = squeeze2x(z); c = squeeze2x(c)
        # apply levels in reverse
        for L in reversed(range(self.squeeze_levels or 1)):
            z, ld = self.levels[L](z, c, reverse=True)
            logdet = logdet + ld
            if self.use_squeeze:
                z = unsqueeze2x(z); c = unsqueeze2x(c)
        return z, logdet

    def sample_and_logq(self, B: int, spatial_shape, c_map: Tensor, device, dtype):
        C, H, W = spatial_shape
        z = torch.randn(B, C, H, W, device=device, dtype=dtype)
        x, logdet = self.forward(z, c_map)
        D = C * H * W
        logpz = -0.5 * (z ** 2).flatten(1).sum(-1) - 0.5 * D * math.log(2.0 * math.pi)
        logq = logpz - logdet
        return x, logq
