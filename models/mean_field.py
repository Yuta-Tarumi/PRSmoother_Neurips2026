import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# Stochastic Lorenz-96 dynamics (your module)
from models.dynamics.Lorenz96 import L96Dynamics
from models.dynamics.Kolmogorov import KolmogorovDynamics  # scaffold only
from models.neural_network_Kolmogorov import ViT2D, ViT2D_for_mixed, ViT2D_for_sparse
from models.neural_network_Lorenz96 import Conv1DRecognition, Conv1DRecognition_Attn

__all__ = ["MFSmoother"]

def build_obs_mask(observation: str,
                   H=128, W=128,
                   hr_box=(0,0,64,64),
                   lr_offset=3, lr_stride=8,
                   device=None) -> torch.Tensor:
    """                                                                                                                                                                                                                  
    Returns a float mask M in {0,1}^{H×W} where M[y,x]=1 iff there is an observation there.               
    - full:         all ones
    - half:         top half
    - sparse/coarse: 16×16 samples placed at [offset::stride], [offset::stride]
    - mixed:        HR box is dense, LR samples outside HR box
    """
    M = torch.zeros((H, W), dtype=torch.float32, device=device)

    if observation in ["full", "full_pm3"]:
        M[:] = 1.0

    elif observation == "half":
        M[:H//2, :] = 1.0

    elif observation in ["coarse", "sparse", "sparse_pm3"]:
        ys = lr_offset + lr_stride * torch.arange(16, device=device)
        xs = lr_offset + lr_stride * torch.arange(16, device=device)
        M[ys.unsqueeze(1), xs.unsqueeze(0)] = 1.0

    elif observation in ["mixed", "quarterHR_LR", "mixed_pm3"]:
        # HR quarter (dense)                                                                                                                                                                                              
        y0, x0, hh, ww = hr_box
        assert (hh, ww) == (64, 64), "This helper assumes 64×64 HR"
        M[y0:y0+hh, x0:x0+ww] = 1.0
        # LR samples outside the HR box                                                                                                                                                                                   
        ys = lr_offset + lr_stride * torch.arange(16, device=device)
        xs = lr_offset + lr_stride * torch.arange(16, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        outside_hr = ~((Y >= y0) & (Y < y0+hh) & (X >= x0) & (X < x0+ww))
        M[Y[outside_hr], X[outside_hr]] = 1.0
    else:
        raise ValueError(f"Unknown observation mode: {observation}")

    return M

# --------------------------- utilities ---------------------------

def diag_gauss_logprob(x: Tensor, mu: Tensor, logstd: Tensor, reduce_dims=True) -> Tensor:
    """log N(x; mu, diag(exp(2*logstd))) summed over last dim.
       Returns shape (...,) if reduce_dims else (..., D)."""
    var = torch.exp(2.0 * logstd)
    quad = (x - mu).pow(2) / var
    log_two_pi = math.log(2.0 * math.pi)
    # sum over last dim
    lp = -0.5 * (quad + log_two_pi).sum(dim=-1) - logstd.sum(dim=-1)
    return lp


def isotropic_gauss_logprob(x: Tensor, mu: Tensor, log_sigma: Tensor) -> Tensor:
    """log N(x; mu, sigma^2 I) summed over last dim (x,mu shape (...,D))."""
    sigma2 = torch.exp(2.0 * log_sigma).to(x.dtype)
    diff2 = (x - mu).pow(2).sum(dim=-1)
    D = x.size(-1)
    return -0.5 * (diff2 / sigma2 + D * math.log(2.0 * math.pi * sigma2))

def isotropic_gauss_logprob_nd(x: Tensor, mu: Tensor, log_sigma: Tensor) -> Tensor:
    """
    Isotropic Gaussian log-prob over ALL dims from dim=2 onward.
    x, mu: shape (B, T-1, H, W) or (B, T-1, D1, ..., Dk)
    log_sigma: scalar (tensor/param or python float)
    Returns: (B, T-1)
    """
    ls = torch.as_tensor(log_sigma, dtype=x.dtype, device=x.device)
    sigma2 = torch.exp(2.0 * ls)
    reduce_dims = tuple(range(2, x.dim()))
    diff2 = (x - mu).pow(2).sum(dim=reduce_dims)
    Dtot = x.flatten(2).size(-1)  # number of per-time dimensions, no data copy
    return -0.5 * (diff2 / sigma2 + Dtot * (math.log(2.0 * math.pi) + 2.0 * ls))


def diag_gauss_logprob_nd(x: Tensor, mu: Tensor, logstd: Tensor) -> Tensor:
    """
    Diagonal Gaussian log-prob over ALL dims from dim=2 onward.
    x, mu, logstd: shape (B, T, H, W) or (B, T, D1, ..., Dk)
    Returns: (B, T)
    """
    var = torch.exp(2.0 * logstd)
    reduce_dims = tuple(range(2, x.dim()))
    quad_sum = ((x - mu).pow(2) / var).sum(dim=reduce_dims)
    Dtot = x.flatten(2).size(-1)
    log_two_pi = math.log(2.0 * math.pi)
    return -0.5 * quad_sum - 0.5 * Dtot * log_two_pi - logstd.sum(dim=reduce_dims)


# ===== deterministic Lorenz-96 step (mean of p(X_t | X_{t-1})) =====
def l96_rhs(x: Tensor, F: Tensor) -> Tensor:
    """RHS for Lorenz-96 with cyclic indexing. x: (B,D)"""
    # roll: x_{i+1}, x_{i-1}, x_{i-2}
    xp1 = torch.roll(x, shifts=-1, dims=-1)
    xm1 = torch.roll(x, shifts=1,  dims=-1)
    xm2 = torch.roll(x, shifts=2,  dims=-1)
    return (xp1 - xm2) * xm1 - x + F  # (B,D)


def l96_integrate_rk4(x0: Tensor, F: Tensor, obs_dt: float, internal_dt: float) -> Tensor:
    """RK4 over substeps from t to t+obs_dt. x0: (B,D)"""
    n_sub = max(1, int(round(obs_dt / internal_dt)))
    dt = obs_dt / n_sub
    x = x0
    for _ in range(n_sub):
        k1 = l96_rhs(x, F)
        k2 = l96_rhs(x + 0.5 * dt * k1, F)
        k3 = l96_rhs(x + 0.5 * dt * k2, F)
        k4 = l96_rhs(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x


# --------------------------- MFSmoother ---------------------------

class MFSmoother(nn.Module):
    """
    Mean-Field Smoother (MFS) implementing q(X_{1:T}) = Π_t N(μ_t, diag(σ_t^2)),
    trained by the ELBO from the mean-field factorization. Compatible with your
    evaluate() function: forward() returns (neg_elbo, xs_sample, elbo1, elbo2, elbo3).

    Supported datasets:
      - "Lorenz96" (D=40) and "Lorenz96_4dim" (D=4)
    """
    def __init__(
        self,
        dataset: str,
        latent_dim: int,
        init_F: float,
        train_F: bool,
        train_Q: bool,
        observation: str,
        norm_factor: float,
        sigma0_sq: float,
        encoder: str,             # kept for API compatibility (not used here)
        steps: int,               # T
        obs_var=1.0,
        learn_obs_var=True,
    ):
        super().__init__()
        assert dataset in ["Lorenz96", "Lorenz96_4dim", "Kolmogorov"]
        self.dataset = dataset
        assert observation in ["full", "half", "mixed", "coarse", "sparse", "square", "biased_abs", "full_pm3", "sparse_pm3", "full_linear", "full_nonlinear"]
        self.observation = observation
        self.latent_dim = latent_dim
        self.T = int(steps)
        self.norm_factor = float(norm_factor)
        self.sigma0_sq = float(sigma0_sq)
        assert self.sigma0_sq > 0

        # ----- observation shape & learnable bias (as in your PRSmoother) -----
        if self.dataset == "Lorenz96":
            if self.observation in ["full", "full_pm3", "square", "biased_abs", "full_linear", "full_nonlinear"]:
                self.obs_dim = 40
                self.bias = nn.Parameter(torch.zeros(40, device="cuda"))
            else:  # sparse
                self.obs_dim = 10
                self.bias = nn.Parameter(torch.zeros(10, device="cuda"))
        elif self.dataset == "Lorenz96_4dim":
            self.obs_dim = 4
            self.bias = nn.Parameter(torch.zeros(4, device="cuda"))
        else:
            # Kolmogorov scaffold only
            if self.observation in ["full", "full_pm3"]:
                self.obs_dim = 128 * 128
                self.bias = nn.Parameter(torch.zeros(128, 128, device="cuda"))
            elif self.observation in ["coarse"]:
                self.obs_dim = 16 * 16
                self.bias = nn.Parameter(torch.zeros(128, 128, device="cuda"))
            elif self.observation in ["half"]:
                self.obs_dim = 64 * 128
                self.bias = nn.Parameter(torch.zeros(64, 128, device="cuda"))
            elif self.observation in ["sparse"]:
                self.bias = nn.Parameter(torch.zeros(128, 128))  # device set via model.to(device)
                # Build and register the observation mask (follows your data layout)
                lr_offset  = getattr(self, "lr_offset", 3)
                lr_stride  = getattr(self, "lr_stride", 8)

                M = build_obs_mask(self.observation,
                                   H=128, W=128,
                                   lr_offset=lr_offset, lr_stride=lr_stride,
                                   device=None)  # will move with model.to(device)
                self.register_buffer("obs_mask", M)            # (128,128), 1=observed, 0=not observed
                self.obs_dim = int(self.obs_mask.sum().item()) # number of sensors (useful for logging)
            elif self.observation in ["mixed"]:
                self.bias = nn.Parameter(torch.zeros(128, 128))  # device set via model.to(device)
                # Build and register the observation mask (follows your data layout)
                hr_box     = getattr(self, "hr_box", (0, 0, 64, 64))
                lr_offset  = getattr(self, "lr_offset", 3)
                lr_stride  = getattr(self, "lr_stride", 8)

                M = build_obs_mask(self.observation,
                                   H=128, W=128,
                                   hr_box=hr_box,
                                   lr_offset=lr_offset, lr_stride=lr_stride,
                                   device=None)  # will move with model.to(device)
                self.register_buffer("obs_mask", M)            # (128,128), 1=observed, 0=not observed
                self.obs_dim = int(self.obs_mask.sum().item()) # number of sensors (useful for logging)        

        # ----- observation noise σ² (scalar) -----
        log_var_init = math.log(obs_var)
        if learn_obs_var:
            self.log_r = nn.Parameter(torch.tensor(log_var_init))
        else:
            self.register_buffer("log_r", torch.tensor(log_var_init))

        # ----- dynamics -----
        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            # reuse the same dynamics container for F and log_sigma
            self.dyn = L96Dynamics(
                init_F=init_F, train_F=train_F, train_Q=train_Q, init_log_sigma=-1.0,
                obs_dt=0.03, internal_dt=0.006,
            )
            # cache for integrator
            self._obs_dt = float(getattr(self.dyn, "obs_dt", 0.03))
            self._internal_dt = float(getattr(self.dyn, "internal_dt", 0.001))
        elif self.dataset == "Kolmogorov":
            # you can extend mean-field to 2D similarly; left as future work
            self.dyn = KolmogorovDynamics(device="cuda", init_log_sigma=1e-2)

        # ----- mean-field posterior head (amortized) -----
        # Simple but effective: BiGRU over time on y_t, linear heads to μ_t, logσ_t.
        if self.dataset in ["Lorenz96"]:
            self.mf_encoder = Conv1DRecognition_Attn(steps=steps, out_channels=100)
        elif self.dataset in ["Lorenz96_4dim"]:
            self.mf_encoder = Conv1DRecognition(steps=steps, out_channels=2*self.T)
            """
            hid = 256
            self.mf_encoder = nn.GRU(
                input_size=self.obs_dim, hidden_size=hid, num_layers=2,
                batch_first=True, bidirectional=True
            )
            self.to_mu     = nn.Linear(2*hid, latent_dim)
            self.to_logstd = nn.Linear(2*hid, latent_dim)
            """
        elif self.dataset in ["Kolmogorov"]:
            # 2D encoder (Kolmogorov)
            self.to_mu_logstd_2d = ViT2D(in_ch=10, out_ch=20)
            if self.observation in ["mixed"]:
                self.to_mu_logstd_2d = ViT2D_for_mixed(in_ch=10, out_ch=20)
            elif self.observation in ["sparse"]:
                self.to_mu_logstd_2d = ViT2D_for_sparse(in_ch=10, out_ch=20)
            else:
                self.to_mu_logstd_2d = ViT2D(in_ch=10, out_ch=20)

            # --- Spectral prior parameters (log-space for positivity) ---
            self.log_alpha = nn.Parameter(torch.tensor(math.log(1e-3)))  # DC floor
            self.log_beta  = nn.Parameter(torch.tensor(math.log(1.0)))   # Laplacian strength
            self.log_gamma = nn.Parameter(torch.tensor(math.log(0.1)))   # bi-Laplacian term
            self.log_raw_boost = nn.Parameter(torch.tensor(math.log(5.0)))
            self.k_cut = 16.0         # cutoff in wavenumber index units (radial)
            self.eta   = 12.0
            self._k2 = None  # will cache (H,W) grid on first call
            self._lam_cache = {}      # keyed by (H, W, device)

    def _get_k2(self, H, W, device, dtype):
        if (self._k2 is None) or (self._k2.shape != (H, W)) or (self._k2.device != device):
            ky = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
            kx = torch.fft.fftfreq(W, d=1.0, device=device).view(1, W)
            self._k2 = (ky**2 + kx**2).to(dtype)
        return self._k2

    def _get_laplacian_eigs(self, H, W, device):
        # cache per (H,W,device) in fp32
        key = (H, W, device)
        lam = self._lam_cache.get(key)
        if lam is None:
            kx = torch.arange(W, device=device, dtype=torch.float32)
            ky = torch.arange(H, device=device, dtype=torch.float32)
            # λ(kx,ky) = 2 - 2 cos(2π kx/W)  +  2 - 2 cos(2π ky/H)
            lam_x = 2.0 - 2.0 * torch.cos(2.0 * math.pi * kx / W)  # (W,)
            lam_y = 2.0 - 2.0 * torch.cos(2.0 * math.pi * ky / H)  # (H,)
            lam = lam_y.view(H, 1) + lam_x.view(1, W)              # (H,W) fp32
            self._lam_cache[key] = lam
        return lam

    def _radial_highk_boost(self, H, W, device, dtype):
        # frequency indices in "index units" (so Nyquist per axis is 64 on 128x128)
        fy = torch.fft.fftfreq(H, d=1.0).to(dtype=dtype, device=device).view(H,1) * H
        fx = torch.fft.fftfreq(W, d=1.0).to(dtype=dtype, device=device).view(1,W) * W
        r  = torch.sqrt(fy*fy + fx*fx)  # (H,W)
        B = 1.0 + torch.exp(self.log_raw_boost)
        # soft wall that turns on beyond k_cut
        sig = torch.sigmoid(torch.as_tensor(self.eta, dtype=dtype, device=device) * (r - self.k_cut))
        return 1.0 + (B - 1.0) * sig  # (H,W)    

    def spectral_log_prior(self, x1):
        """
            Spectral Gaussian prior for the first state.
            x1: (B,H,W) or (B,C,H,W). Sums over channels if present.
            log p = -0.5 * sum_k [ (α + βλ + γλ^2) * |X̂(k)|^2 ]  + 0.5 * sum_k log(α + βλ + γλ^2)  - 0.5 * D log(2π)
        """
        # Sum per-channel if provided
        if x1.dim() == 4:
            return sum(self.spectral_log_prior(x1[:, c]) for c in range(x1.size(1)))
        
        B, H, W = x1.shape
        D = H * W
        two_pi_log = math.log(2.0 * math.pi)
        
        out_dtype = x1.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x = x1.float()                                # fp32 for stability
            X = torch.fft.fft2(x, norm="ortho")           # unitary DFT
            lam = self._get_laplacian_eigs(H, W, x.device) # (H,W) fp32
            alpha = self.log_alpha.exp().float()
            beta  = self.log_beta.exp().float()
            gamma  = self.log_gamma.exp().float()
            base_power = alpha + beta*lam + gamma*(lam**2)
            # learnable top-octave wall
            Bwall = self._radial_highk_boost(H, W, x.device, base_power.dtype)  # (H,W)                                                                                                                                  
            power = base_power * Bwall
            quad  = (power * (X.real**2 + X.imag**2)).sum(dim=(-1,-2))
            const = 0.5 * (power.clamp_min(1e-12).log().sum()) - 0.5 * D * math.log(2*math.pi)
            lp    = (-0.5 * quad + const).to(out_dtype)
            """
            lam = self._get_laplacian_eigs(H, W, x.device)  # (H,W) fp32
            alpha = self.log_alpha.exp().float()
            beta  = self.log_beta.exp().float()
            gamma = self.log_gamma.exp().float()
            
            power = alpha + beta * lam + gamma * (lam ** 2)           # (H,W) > 0
            # quadratic form via Parseval (unitary): sum power * |X̂|^2
            quad = (power * (X.real * X.real + X.imag * X.imag)).sum(dim=(-1, -2))  # (B,)
            
            # log-normalizer
            const = 0.5 * power.log().sum() - 0.5 * D * two_pi_log     # scalar
                
            lp = (-0.5 * quad + const).to(out_dtype)                   # (B,)
            """
            return lp

     # inside your model class (the one that has self.bias and self.obs_mask)                                                                                                                                              
    def add_masked_bias(self, Hx_full: torch.Tensor) -> torch.Tensor:
        """                                                                                                                                                                                                              
        Hx_full: (B,T,128,128) predicted observation on the full canvas.                                                                                                                                                 
        Returns Hx_full + bias only at observed pixels.                                                                                                                                                                  
        """
        # Shapes: obs_mask -> (128,128)  -> broadcast to (B,T,128,128)
        #         bias     -> (128,128)  -> broadcast to (B,T,128,128)                                                                                                                                                   
        M = self.obs_mask[None, None, :, :]                  # (1,1,128,128)
        return Hx_full + self.bias[None, None, :, :] * M

    def flatten_observed(self, x_full: torch.Tensor) -> torch.Tensor:
        """                                                                                                                                                                                                              
        x_full: (B,T,128,128)  -> (B,T,obs_dim) by selecting observed pixels.                                                                                                                                            
        """
        B, T, H, W = x_full.shape
        Mbt = self.obs_mask.bool()[None, None, :, :].expand(B, T, H, W)
        return x_full.masked_select(Mbt).view(B, T, -1)
    
    # ---------------- observation log-prob (reused logic) ----------------
    def obs_log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Log p(y|x) under diagonal Gaussian with variance exp(log_r).
        L96 case: y,x shapes (B,T,D). Returns (B,T).
        """
        var = torch.exp(self.log_r).to(y.dtype)  # scalar
        two_pi = torch.tensor(2.0 * math.pi, dtype=y.dtype, device=y.device)

        if x.dim() == 3 and y.dim() == 3:
            if self.observation == "full_linear":
                diff = y - x
                #diff = y - (x + self.bias)
            elif self.observation == "full_nonlinear":
                #x_to_4_clamped = torch.clamp((x+self.bias)**4, 0, 10)
                x_to_2_clamped = torch.clamp(x**2, 0, 5)
                diff = y - x_to_2_clamped
            elif self.observation == "full_pm3":
                diff = y - torch.clamp(x + self.bias, -3.0, 3.0)
            elif self.observation == "square":
                diff = y - (x * x + self.bias)
            elif self.observation == "biased_abs":
                diff = y - (torch.abs(x - 2) + self.bias)
            elif self.observation == "sparse":
                diff = y - (x[..., ::4] + self.bias)
            elif self.observation == "sparse_pm3":
                diff = y - torch.clamp(x[..., ::4] + self.bias, -3.0, 3.0)
            else:
                raise ValueError(f"Unknown observation mode {self.observation}")
            return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(-1)
        elif x.dim() == 4 and y.dim() == 4:
            # x: (B,T,Hx,Wx), y: (B,T,Hy,Wy)
            Hx, Wx = x.shape[-2], x.shape[-1]
            Hy, Wy = y.shape[-2], y.shape[-1]
            if self.observation in ["full", "full_pm3"]:
                x_for_y = x+self.bias
                if self.observation.endswith("pm3"):
                    x_for_y = torch.clamp(x_for_y, -3.0, 3.0)
                diff = y - x_for_y
            elif self.observation in ["half"]:
                x_for_y = x[:, :, :64, :]+self.bias
                if self.observation.endswith("pm3"):
                    x_for_y = torch.clamp(x_for_y, -3.0, 3.0)
                diff = y - x_for_y
            elif self.observation in ["coarse"]:
                x_for_y = F.avg_pool2d(x, kernel_size=8, stride=8)+self.bias
                diff = y - x_for_y
            elif self.observation in ["sparse_pm3"]:
                assert Hx % Hy == 0 and Wx % Wy == 0, "x and y must be integer-factor related"
                sh, sw = Hx // Hy, Wx // Wy
                x_ds = x[..., ::sh, ::sw]  # (B,T,Hy,Wy)                                                                                                                                      
                if self.observation.endswith("pm3"):
                    x_ds = torch.clamp(x_ds, -3.0, 3.0)
                diff = y - x_ds
            elif self.observation in ["mixed", "mixed_pm3", "sparse"]:
                # Requires: self.obs_mask (128x128 buffer), self.bias (128x128 param),
		# and the helpers: add_masked_bias(...), flatten_observed(...)
                assert hasattr(self, "obs_mask"), "obs_mask not set for 'mixed' mode"
                assert (Hx, Wx) == (128, 128) and (Hy, Wy) == (128, 128), \
                    "mixed expects full 128x128 canvases with NaNs at unobserved pixels"

                # Add bias only at observed pixels, then (optionally) clamp
                x_full_biased = self.add_masked_bias(x)  # (B,T,128,128)
                if self.observation.endswith("pm3"):
                    x_full_biased = torch.clamp(x_full_biased, -3.0, 3.0)

                # Select only observed entries (mask-driven); NaNs in y are ignored
                y_vec  = self.flatten_observed(torch.nan_to_num(y))       # (B,T,obs_dim)
                x_vec  = self.flatten_observed(x_full_biased)             # (B,T,obs_dim)
                diff_vec = y_vec - x_vec
                return -0.5 * ((diff_vec ** 2) / var + torch.log(two_pi * var)).sum(-1)
            else:
                raise ValueError(f"Unknown observation mode {self.observation}")
            return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(dim=(-1, -2))

        raise ValueError(f"Unexpected x,y shapes: x{tuple(x.shape)} y{tuple(y.shape)}")

    # ---------------- amortized mean-field params ----------------
    def _mf_params(self, y: Tensor):
        if y.dim() == 4:
            B, T, H, W = y.shape
            HWmax = np.fmax(H, W)
            print(f"{HWmax=}")
            assert T == self.T, f"Expected T={self.T}, got {T}"
            y_scaled = (y / self.norm_factor)                    # (B,T,H,W)
            mu_2d_logstd_2d = self.to_mu_logstd_2d(y_scaled).squeeze(1)
            mu_2d = mu_2d_logstd_2d[:, :10]
            logstd_2d = mu_2d_logstd_2d[:, 10:]
            mu     = mu_2d.view(B, T, HWmax, HWmax)                      # (B,T,H,W)
            logstd = logstd_2d.view(B, T, HWmax, HWmax).clamp(-7.0, 5.0)
            return mu, logstd
        else:
            y_scaled = y / self.norm_factor
            B, T, _ = y_scaled.shape
            assert T == self.T, f"Expected T={self.T}, got {T}"
            mu_logstd = self.mf_encoder(y_scaled)
            mu = mu_logstd[:, :T]
            logstd = mu_logstd[:, T:]
            #feat, _ = self.mf_encoder(y_scaled)              # (B,T,2*hid)
            #mu      = self.to_mu(feat)                       # (B,T,D)
            #logstd  = self.to_logstd(feat).clamp(-7.0, 5.0)  # stability
            return mu, logstd

    # ---------------- dynamics mean: f(X_{t-1}) ----------------
    def _mean_step(self, x_prev: Tensor) -> Tensor:
        """
        Deterministic mean of p(X_t|X_{t-1}) for L96 via RK4 (float32 for stability).
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = x_prev.float()
            if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
                # F may be a Parameter/buffer; broadcast to (B,D)
                """
                F_scalar = getattr(self.dyn, "F")
                F = torch.as_tensor(F_scalar, dtype=x.dtype, device=x.device)
                F = F.expand_as(x)
                x_next = l96_integrate_rk4(x, F, self._obs_dt, self._internal_dt)
                """
                x_next = self.dyn(x_prev)
                assert torch.isfinite(x_next).all()
                return x_next.to(x_prev.dtype)
            elif self.dataset in ["Kolmogorov"]:
                x_next = self.dyn(x_prev)
                return x_next.to(x_prev.dtype)
            else:
                raise NotImplementedError("Mean-step not implemented.")

    # ---------------- forward / ELBO ----------------
    def forward(self, y: Tensor):
        """
        Inputs:
          - L96: y shape (B,T,D_obs)
        Returns:
          - neg_elbo, xs (sampled from q_t), elbo1, elbo2, elbo3
            where elbo2 includes (obs + transition), elbo3 is Σ_t E log q_t.
        """
        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            B, T, d_obs = y.shape
            if self.observation in ["full_linear", "full_nonlinear", "full_pm3", "square", "biased_abs"]:
                assert T == self.T and d_obs == self.obs_dim
            elif self.observation in ["sparse", "sparse_pm3"]:
                assert T == self.T and d_obs == self.obs_dim
        elif self.dataset in ["Kolmogorov"]:
            B, T, d_obs, d_obs = y.shape
            if self.observation in ["full", "full_pm3"]:
                assert T == 10 and d_obs*d_obs == self.latent_dim, "Unexpected input shape"
            elif self.observation in ["half", "coarse", "mixed", "sparse"]:
                assert T == 10, "Unexpected input shape"
            elif self.observation in ["sparse_pm3"]:
                assert T == 10 and d_obs*d_obs == int(self.latent_dim)//128, "Unexpected input shape"

        # 1) amortized mean-field params and reparameterized samples
        mu, logstd = self._mf_params(y)                   # (B,T,D), (B,T,D)
        eps = torch.randn_like(mu)
        xs = mu + torch.exp(logstd) * eps               # (B,T,D)

        # 2) terms of the ELBO
        # prior on X1 (isotropic Gaussian with variance sigma0_sq)
        x1 = xs[:, 0]                                    # (B,D)
        if self.dataset == "Kolmogorov":
            # --- Spectral prior only for Kolmogorov (first timestep) ---
            assert x1.dim() == 3, f"Expected (B,H,W) for Kolmogorov x1, got {tuple(x1.shape)}"
            log_p_x1 = self.spectral_log_prior(x1)  # (B,)
        elif self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            # --- Original isotropic Gaussian prior for L96 paths (unchanged) ---
            x1_flat = x1.reshape(B, -1).to(xs.dtype)
            D_eff = x1_flat.size(-1)
            log_p_x1 = -0.5 * (
                (x1_flat.pow(2).sum(-1) / self.sigma0_sq)
                + D_eff * (math.log(2.0 * math.pi) + math.log(self.sigma0_sq))
            )  # (B,)

        # obs term: sum_t E log p(y_t | x_t)
        lp_obs_bt = self.obs_log_prob(y.to(xs.dtype), xs)  # (B,T)
        lp_obs = lp_obs_bt.sum(dim=1)                      # (B,)

        # transition term: sum_{t>=2} E log p(x_t|x_{t-1})
        # --- transition term: sum_{t>=2} E log p(x_t|x_{t-1}) ---
        if self.dataset in  ["Lorenz96", "Lorenz96_4dim"]:
            B, T, D = xs.shape
        elif self.dataset in ["Kolmogorov"]:
            B, T, H, W = xs.shape
        dyn_log_sigma = getattr(self.dyn, "log_sigma")
        
        if T > 1:
            if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
                x_prev = xs[:, :-1, :]  # (B, T-1, D)
                # mean f(x_{t-1}) via deterministic L96 integration (keep grads for F!)
                mean_next = self._mean_step(x_prev.reshape(-1, D)).view(B, T-1, D)  # (B, T-1, D)
            
                lp_trans = isotropic_gauss_logprob(
                    xs[:, 1:, :],   # x_t
                    mean_next,      # f(x_{t-1})
                    dyn_log_sigma
                ).sum(dim=1)        # (B,)
            elif self.dataset in ["Kolmogorov"]:
                # Keep 4D for dynamics; no flattening for the Gaussian
                x_prev_img    = xs[:, :-1, :, :]                            # (B, T-1, H, W)
                mean_next_img = self._mean_step(x_prev_img.reshape(-1, H, W))  # (B*(T-1), H, W)
                mean_next_img = mean_next_img.view(B, T-1, H, W)             # (B, T-1, H, W)
                
                if dyn_log_sigma is None:
                    # Fallback if your Kolmogorov dynamics doesn’t define log_sigma
                    dyn_log_sigma = xs.new_tensor(-5.0)  # log σ ≈ -5
                    
                # ND isotropic Gaussian: returns (B, T-1) without flattening
                lp_trans = isotropic_gauss_logprob_nd(
                    xs[:, 1:, :, :], mean_next_img, dyn_log_sigma
                ).sum(dim=1)
        else:
            # No transition term when T==1
            lp_trans = xs.new_zeros(B)

        # entropy (sum E log q_t) using sampled xs
        if xs.dim() == 3:
            log_q_bt = diag_gauss_logprob(xs, mu, logstd)     # (B,T)
        elif xs.dim() == 4:
            log_q_bt = diag_gauss_logprob_nd(xs, mu, logstd)     # (B,T)
        log_q_sum = log_q_bt.sum(dim=1)                   # (B,)

        # 3) assemble ELBO pieces
        elbo1 = log_p_x1.mean()                           # prior
        elbo2 = (lp_obs + lp_trans).mean()                # obs + transition
        elbo3 = log_q_sum.mean()                          # sum E log q_t
        elbo  = elbo1 + elbo2 - elbo3
        assert torch.isfinite(lp_obs.mean())
        assert torch.isfinite(lp_trans.mean())
        assert torch.isfinite(xs).all(), "xs NaN/Inf"
        assert torch.isfinite(elbo1), "elbo1 NaN/Inf"
        assert torch.isfinite(elbo2), "elbo2 NaN/Inf"
        assert torch.isfinite(elbo3), "elbo3 NaN/Inf"

        return -elbo, xs, elbo1.detach(), elbo2.detach(), elbo3.detach()
