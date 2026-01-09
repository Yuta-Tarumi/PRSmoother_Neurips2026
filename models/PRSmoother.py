import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import time
import numpy as np

# Stochastic Lorenz-96 dynamics
from models.dynamics.Lorenz96 import L96Dynamics  # updated import path
from models.dynamics.Kolmogorov import KolmogorovDynamics
# Periodic convolutional recognition network
from models.neural_network_Lorenz96 import LinearRecognition, Conv1DRecognition, Conv1DRecognition_Attn, FCRecognition, Transformer
from models.neural_network_Kolmogorov import Conv2DRecognition, ViT2D, ViT2D_for_mixed, ViT2D_for_sparse
from models.flow import RealNVP_Flow1D, Spline_Flow1D, _Flow2D, GaussianShockHead, GaussianShockHead2D, NoisyGaussianB1, TransformerFutureEncoder1D
__all__ = ["PRSmoother"]

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

    if observation in ["full", "full_pm3", "full_linear", "full_nonlinear"]:
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


class GaussianShockHead1D_TrainableLogStd(nn.Module):
    """1‑D Gaussian shock head with an *external* (trainable) log‑std.
    
    This is the minimal change needed to make the variational transition
    covariance S_t trainable in PRSmoother, while keeping the process noise
    covariance Q separate.

    The head predicts only the mean correction m_t(c), and uses
    `logstd_fn()` to supply log σ (scalar or per‑dimension). The resulting
    q(δ_t|c) is diagonal Gaussian with std = exp(log σ).
    """

    def __init__(
        self,
        D: int,
        context_dim: int,
        logstd_fn,
        hidden: int = 512,
        mean_bound: float = 0.5,
    ):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, D),
        )
        self.mean_bound = float(mean_bound)
        self._logstd_fn = logstd_fn  # callable returning ( ) or (D,)

    def forward(self, c: Tensor):
        m = self.mean_bound * torch.tanh(self.mu(c))

        ls = self._logstd_fn()
        if not isinstance(ls, torch.Tensor):
            ls = torch.as_tensor(ls, device=m.device, dtype=m.dtype)
        ls = ls.to(device=m.device, dtype=m.dtype)

        # broadcast scalar → (B,D) or vector (D,) → (B,D)
        if ls.ndim == 0:
            ls = ls.expand_as(m)
        else:
            ls = ls.view(1, -1).expand_as(m)
        return m, ls

# --------------------------- PRSmoother ---------------------------

class PRSmoother(nn.Module):
    """Physics-Rollout Smoother (PRS) with pluggable periodic-Conv encoder.

    Default encoder: eight-layer Conv1d stack (see `Conv1DRecognition`) with
    circular padding, mapping an observation cube of shape (B, 50, 40) – where
    50 is the time window – to a latent feature (B, 1, 40).  Dynamics has only
    two learnable parameters: scalar forcing **F** and scalar `log_sigma`.
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
        encoder: str,
        steps: int,
        obs_var=1.0,
        learn_obs_var=True,
        posterior: str = "flow",
        noisy=False,
        log_dyn_noise=torch.tensor(math.log(0.01)), # used in the noisy path
        log_gen_dyn_noise=torch.tensor(math.log(0.01)), # used in the noisy path
    ):
        super().__init__()
        assert dataset in ["Lorenz96", "Lorenz96_4dim", "Kolmogorov"]
        self.dataset = dataset
        assert observation in ["full", "half", "coarse", "mixed", "sparse", "square", "biased_abs", "full_pm3", "sparse_pm3", "full_linear", "full_nonlinear"]
        self.observation = observation
        print(f"{self.observation=}")
        if self.dataset in ["Lorenz96"]:
            if self.observation in ["full_linear", "full_pm3", "square", "biased_abs", "full_nonlinear"]:
                obs_dim = 40
                self.bias = nn.Parameter(torch.zeros(40, device="cuda"))
            elif self.observation in ["sparse", "sparse_pm3"]:
                obs_dim = 10
                self.bias = nn.Parameter(torch.zeros(10, device="cuda"))
        elif self.dataset in ["Lorenz96_4dim"]:
            obs_dim = 4
            self.bias = nn.Parameter(torch.zeros(4, device="cuda"))
        elif self.dataset in ["Kolmogorov"]:
            if self.observation in ["full", "full_pm3"]:
                obs_dim = 128*128
                self.bias = nn.Parameter(torch.zeros(128, 128, device="cuda"))
            elif self.observation in ["half"]:
                obs_dim = 64*128
                self.bias = nn.Parameter(torch.zeros(64, 128, device="cuda"))
            elif self.observation in ["coarse", "sparse_pm3"]:
                obs_dim = 16*16
                self.bias = nn.Parameter(torch.zeros(16, 16, device="cuda"))
            elif self.observation in ["sparse"]:
                self.bias = nn.Parameter(torch.zeros(128, 128))  # device set via model.to(device)
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
        self.norm_factor = float(norm_factor)
        self.latent_dim = latent_dim
        self.sigma0_sq = sigma0_sq
        assert self.sigma0_sq > 0
        self.posterior = posterior
        self.noisy = noisy
        # --- Trainable variational transition noise (affects S_t ONLY, not Q) ---
        # We want log_dyn_noise <= 0 (i.e., std <= 1). Parameterize via
        #   log_dyn_noise = log(sigmoid(raw)) = logsigmoid(raw) in (-inf, 0).
        # This keeps the *maximum* at 0 while leaving the minimum unbounded.
        log_dyn_noise_init = torch.as_tensor(log_dyn_noise, dtype=torch.float32)
        log_gen_dyn_noise_init = torch.as_tensor(log_gen_dyn_noise, dtype=torch.float32)
        # Safety: if caller passes a positive init, clip to 0 so std<=1 holds.
        log_dyn_noise_init = torch.minimum(log_dyn_noise_init, torch.zeros_like(log_dyn_noise_init))
        log_gen_dyn_noise_init = torch.minimum(log_gen_dyn_noise_init, torch.zeros_like(log_gen_dyn_noise_init))
        p = log_dyn_noise_init.exp().clamp(1e-6, 1.0 - 1e-6)  # p in (0,1)
        p_gen = log_gen_dyn_noise_init.exp().clamp(1e-6, 1.0 - 1e-6)  # p in (0,1)
        raw_init = torch.log(p) - torch.log1p(-p)             # logit(p)
        raw_init_gen = torch.log(p_gen) - torch.log1p(-p_gen)
        self._log_dyn_noise_raw = nn.Parameter(raw_init)
        self._log_gen_dyn_noise_raw = nn.Parameter(raw_init_gen)
        # (Optional) print once for debugging; safe for scalar init.
        try:
            print(f"log_dyn_noise(init)={float(self.log_dyn_noise.detach().cpu()):.4g} (constrained <= 0)")
        except Exception:
            print("log_dyn_noise initialized (constrained <= 0)")
        assert self.posterior in {"flow", "flow_Spline", "flow_RealNVP", "diag_gauss"}, f"Unknown posterior={posterior}"

        # ── observation noise σ² (scalar) ────────────────────────────
        log_var_init = math.log(obs_var)
        if learn_obs_var:
            self.log_r = nn.Parameter(torch.tensor(log_var_init))
        else:
            self.register_buffer("log_r", torch.tensor(log_var_init))

        # ───── Dynamics block ───────────────────────────────────────────
        if self.dataset in ["Lorenz96"]:
            self.dyn = L96Dynamics(
                init_F=init_F, train_F=train_F, train_Q=train_Q, init_log_sigma=-100.0,
                obs_dt=0.03, internal_dt=0.006,
            )
        elif self.dataset in ["Lorenz96_4dim"]:
            self.dyn = L96Dynamics(
                init_F=init_F, train_F=train_F, train_Q=train_Q, init_log_sigma=np.log(1e-8),
                obs_dt=0.03, internal_dt=0.006,
            )
        elif self.dataset in ["Kolmogorov"]:
            self.dyn = KolmogorovDynamics(
                device="cuda", internal_dt=5.0e-3
            )
        else:
            print("Unknwon dataset: {self.dataset=}")
            return 1

        # --- Process noise (Q) for the *generative* dynamics prior p_theta_dyn ---
        # This is distinct from `log_dyn_noise`, which parameterizes the variational
        # transition covariance S_t (see Sec. 2.4.3 / Eq. (14) in the paper).
        #
        # Lorenz96Dynamics already exposes a (trainable) log_sigma for Q. KolmogorovDynamics
        # in this codebase is deterministic, so we create a separate scalar log_sigma buffer/
        # parameter for the prior term only.
        if not hasattr(self.dyn, "log_sigma"):
            log_sigma_init = math.log(1e-8)
            if train_Q:
                self.log_sigma = nn.Parameter(torch.tensor(log_sigma_init, dtype=torch.float32))
            else:
                self.register_buffer("log_sigma", torch.tensor(log_sigma_init, dtype=torch.float32))

        # ───── Recognition network q_φ context encoder ────────────────
        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            assert encoder in ["Conv1D", "Conv1D_flow", "Linear_flow", "Transformer", "ConvTransformer", "FC"]
            if encoder == "Conv1D":
                if self.dataset in ["Lorenz96"]:
                    #self.encoder = Conv1DRecognition(steps=steps)   # (B,50,40) → (B,2,40)
                    self.encoder = Conv1DRecognition_Attn(steps=steps)
                else:
                    self.encoder = Conv1DRecognition(steps=steps)
            elif encoder == "Conv1D_flow":
                #self.encoder = Conv1DRecognition(steps=steps, out_channels=64)   # (B,50,40) → (B,2,40)
                if self.dataset in ["Lorenz96"]:
                    self.encoder = Conv1DRecognition_Attn(steps=steps, out_channels=64)
                else:
                    self.encoder = Conv1DRecognition(steps=steps, out_channels=64)

            elif encoder == "Linear_flow":
                #self.encoder = Conv1DRecognition(steps=steps, out_channels=64)   # (B,50,40) → (B,2,40)
                if self.dataset in ["Lorenz96"]:
                    print("not yet")
                    return 1
                else:
                    self.encoder = LinearRecognition(steps=steps, out_channels=64)

            if self.posterior in ["flow_RealNVP"]:
                # richer context to drive the flow
                self._flow_ctx_dim = 256
                self.flow_ctx_proj = nn.Sequential(
                    nn.Linear(64 * self.latent_dim, 256), nn.SiLU(),
                    nn.Linear(256, self._flow_ctx_dim),
                )

                # upgraded 1D flow (more layers, deeper coupling, mixing, actnorm)
                if self.noisy:
                    print("flow for the noisy mode (Option B1: states-out, shocks-inside)")
                    Cs_dim = 128
                    self.future_ctx = TransformerFutureEncoder1D(
                        d_in=self.latent_dim, d_model=256, nhead=8, layers=8, ff=512,
                        Cg=self._flow_ctx_dim, Cs=Cs_dim
                    )
                    flow_x1 = RealNVP_Flow1D(D=self.latent_dim, context_dim=self._flow_ctx_dim, n_layers=8, hidden=512,
                                             n_reflections=8, scale_clamp=5.0, use_actnorm=True)
                    # Variational transition q(delta_t | x_t, y_{t+1:T}) uses S_t.
                    # We make its log-std trainable in PRSmoother (log_dyn_noise) with an upper bound at 0.
                    shock_head = GaussianShockHead1D_TrainableLogStd(
                        D=self.latent_dim,
                        context_dim=self.latent_dim + Cs_dim + 1,
                        logstd_fn=(lambda: self.log_dyn_noise),
                        hidden=512,
                        mean_bound=0.5,
                    )
                    
                    # IMPORTANT: Q (process noise covariance in the generative model) is a separate parameter.
                    # PRSmoother.forward computes the dynamics prior term using the dynamics module; the
                    # NoisyGaussianB1 "Q" is only used for its auxiliary logp_dyn_sum (not required by the trainer).
                    dev = self.bias.device if hasattr(self, "bias") else None
                    Q_init = torch.eye(self.latent_dim, device=dev)
                    self.flow = NoisyGaussianB1(
                        D=self.latent_dim,
                        steps=steps,
                        f_theta=self.dyn,
                        flow_x1=flow_x1,
                        shock_head=shock_head,
                        Q=Q_init,
                    )
                    
                else:
                    self.flow = RealNVP_Flow1D(
                        D=self.latent_dim,
                        context_dim=self._flow_ctx_dim,
                        n_layers=8,
                        hidden=256,
                        n_reflections=8,
                        scale_clamp=5.0,
                        use_actnorm=True,
                    )
            elif self.posterior in ["flow_Spline"]:
                # richer context to drive the flow
                self._flow_ctx_dim = 256
                self.flow_ctx_proj = nn.Sequential(
                    nn.Linear(64 * self.latent_dim, 256), nn.SiLU(),
                    nn.Linear(256, self._flow_ctx_dim),
                )

                # upgraded 1D flow (more layers, deeper coupling, mixing, actnorm)
                self.flow = Spline_Flow1D(
                    D=self.latent_dim,
                    context_dim=self._flow_ctx_dim,
                    n_layers=8,
                    hidden=256,
                    n_reflections=8,
                    scale_clamp=5.0,
                    use_actnorm=True,
                )
            elif self.posterior in ["diag_gauss"]:
                if self.noisy:
                    print("diag_gauss for the noisy mode (Option B1: states-out, shocks-inside)")
                    Cs_dim = 128
                    # If you already have self._flow_ctx_dim elsewhere, keep it; otherwise provide a default
                    if not hasattr(self, "_flow_ctx_dim"):
                        self._flow_ctx_dim = 256

                    self.future_ctx = TransformerFutureEncoder1D(
                        d_in=self.latent_dim, d_model=256, nhead=8, layers=2, ff=512,
                        Cg=self._flow_ctx_dim,  # c_global will be produced but not used by the diag-Gauss path
                        Cs=Cs_dim
                    )

                    # 2) shock head for per-step diagonal-Gaussian model errors
                    #    context = [current state (D), per-step features (Cs_dim), time_frac (1)] → total D + Cs_dim + 1
                    self.shock_head = GaussianShockHead(
                        D=self.latent_dim,
                        context_dim=self.latent_dim + Cs_dim + 1,
                        hidden=256
                    )

                    # 3) minimal wrapper so code can call self.flow.{f_theta, shock_head}
                    class _NoisyHeadWrapper(nn.Module):
                        def __init__(self, f_theta, shock_head):
                            super().__init__()
                            self.f_theta = f_theta
                            self.shock_head = shock_head

                    # make it available under self.flow so sample_x1T_diag_gauss can reuse the same attribute names
                    self.flow = _NoisyHeadWrapper(f_theta=self.dyn, shock_head=self.shock_head)

                    # (Optional) if some other code may accidentally touch self.flow_ctx_proj, make it a no-op:
                    if not hasattr(self, "flow_ctx_proj") or self.flow_ctx_proj is None:
                        self.flow_ctx_proj = nn.Identity()
                else:
                    self.flow = None
                    self.flow_ctx_proj = None
                
        elif self.dataset in ["Kolmogorov"]:
            assert encoder in ["Conv2D", "VisionTransformer", "ConvTransformer"]
            if encoder == "Conv2D":
                self.encoder = Conv2DRecognition(kernel_size=3, n_frames=10)
            elif encoder == "VisionTransformer":
                if self.observation in ["mixed"]:
                    self.encoder = ViT2D_for_mixed(in_ch=10)
                elif self.observation in ["sparse"]:
                    self.encoder = ViT2D_for_sparse(in_ch=10)
                else:
                    self.encoder = ViT2D(in_ch=10)
                
            if self.posterior in ["flow"]:
                # 2D flow at 128x128
                self._ctx_ch = 64
                self._ctx_local_ch = 2
                self.ctx_conv = nn.Sequential(
                    nn.Conv2d(2, 64, 1), nn.SiLU(),
                    nn.Conv2d(64, self._ctx_ch, 1),
                )
                self.ctx_local_conv = nn.Sequential(
                    nn.Conv2d(2, 64, 1), nn.SiLU(),
                    nn.Conv2d(64, self._ctx_local_ch, 1),
                )
                self.flow = _Flow2D(in_ch=1, ctx_ch=self._ctx_ch,
                                    use_squeeze=True, squeeze_levels=4,
                                    blocks_per_level=(4, 4, 4, 4),
                                    use_1x1=True, scale_clamp=1.0)

                ## after: self.flow = _Flow1D(...) or _Flow2D(...)
                for mod in self.flow.modules():
                    if hasattr(mod, "net"):
                        last = mod.net[-1]
                        nn.init.zeros_(last.weight)
                        nn.init.zeros_(last.bias)

                if self.noisy:
                    in_ch = 1 + self._ctx_local_ch + 1     # [state, per‑step ctx, time]
                    self.shock_head_2d = GaussianShockHead2D(in_ch=in_ch, hidden=64, mean_bound=0.5, fixed_logstd=-3.0)

            elif self.posterior in ["diag_gauss"]:
                self.flow = None
                self._ctx_ch = 1 # just random
                self.ctx_conv = None
                if self.noisy:
                    # we’ll reuse the encoder directly per step, so no global ctx_conv here
                    self._ctx_local_ch = 2
                    self.ctx_local_conv = nn.Sequential(
                        nn.Conv2d(2, 64, 1), nn.SiLU(),
                        nn.Conv2d(64, self._ctx_local_ch, 1),
                    )
                    in_ch = 1 + self._ctx_local_ch + 1
                    self.shock_head_2d = GaussianShockHead2D(in_ch=in_ch, hidden=64, mean_bound=0.5, fixed_logstd=-3.0)
        # in __init__ (no fixed size; we'll cache per H,W) : for spectral prior
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1e-3)))  # DC floor
        self.log_beta  = nn.Parameter(torch.tensor(math.log(1.0)))   # Laplacian strength
        self.log_gamma = nn.Parameter(torch.tensor(math.log(0.1)))   # bi-laplacian term
        self.log_raw_boost = nn.Parameter(torch.tensor(math.log(5.0)))  
        # init so B≈1+softplus(raw_boost) ≈ 5   (start gentle; you can raise later)
        self.k_cut = 16.0         # cutoff in wavenumber index units (radial)
        self.eta   = 12.0 
        self._k2 = None  # will cache (H,W) grid on first call

    @property
    def log_dyn_noise(self) -> Tensor:
        """Trainable log-std for the variational transition covariance S_t.
        
        This parameter is used for S_t in the noisy variational family
        q(X_t|X_{t-1},Y_{t:T}) = N(f(X_{t-1}) + m_t, S_t).

        It is constrained to be <= 0 via logsigmoid(raw), so its maximum is 0.
        """
        return F.logsigmoid(self._log_dyn_noise_raw)

    @property
    def _log_process_noise_std(self) -> Tensor:
        """Return log σ for the *process* noise covariance Q in p(x_{t+1}|x_t).
        
        Lorenz96Dynamics defines `self.dyn.log_sigma`. For Kolmogorov we create
        `self.log_sigma` in PRSmoother.
        """
        #if hasattr(self.dyn, "log_sigma"):
        #    return self.dyn.log_sigma
        #log_sigma = torch.log(torch.tensor(0.1)) # fixed at 0.1
        return F.logsigmoid(self._log_gen_dyn_noise_raw)#log_sigma

    def _build_future_ctx_maps(self, y_scaled: Tensor):
        """
        y_scaled: (B, T, 128, 128)
        Returns:
        c_global: (B, Cc, H, W) from all Y_{1:T}
        c_steps : (B, T-1, Cc, H, W) where step t uses only Y_{t+1:T}
        """
        #assert self.dataset in ["Kolmogorov"] and self._ctx_ch is not None
        B, T, H, W = y_scaled.shape
        
        # global context from all frames
        h_full = self.encoder(y_scaled)            # (B, 2, H, W)
        if h_full.dim() == 3:                      # safety (flattened case)
            B2, C2, D2 = h_full.shape
            HW = int(math.isqrt(D2)); assert HW * HW == D2
            h_full = h_full.view(B2, C2, HW, HW)
        if self.posterior in ["diag_gauss"]:
            c_global = torch.ones(B, self._ctx_ch, H, W)
        else:
            c_global = self.ctx_conv(h_full)           # (B, _ctx_ch, H, W)
            
        # future‑only per‑step maps: zero out past frames (1..t), keep Y_{t+1:T}
        c_steps = []
        for t in range(T - 1):
            y_mask = y_scaled.clone()
            if t + 1 > 1:
                y_mask[:, :t+1, :, :] = 0.0       # keep only Y_{t+1:T}
            h_t = self.encoder(y_mask)            # (B, 2, H, W)
            if h_t.dim() == 3:
                B2, C2, D2 = h_t.shape
                HW = int(math.isqrt(D2)); assert HW * HW == D2
                h_t = h_t.view(B2, C2, HW, HW)
            c_t = self.ctx_local_conv(h_t)              # (B, _ctx_local_ch, H, W)
            c_steps.append(c_t)
        c_steps = torch.stack(c_steps, dim=1)      # (B, T-1, _ctx_local_ch, H, W)
        return c_global, c_steps

    def _get_k2(self, H, W, device, dtype):
        if (self._k2 is None) or (self._k2.shape != (H, W)) or (self._k2.device != device):
            ky = torch.fft.fftfreq(H, d=1.0, device=device).view(H, 1)
            kx = torch.fft.fftfreq(W, d=1.0, device=device).view(1, W)
            self._k2 = (ky**2 + kx**2).to(dtype)
        return self._k2

    def _get_laplacian_eigs(self, H, W, device):
        # cache per (H,W,device) in fp32
        key = (H, W, device)
        if not hasattr(self, "_lam_cache"):
            self._lam_cache = {}
        lam = self._lam_cache.get(key)
        if lam is None:
            # λ(kx,ky) = 4 sin^2(π kx/W) + 4 sin^2(π ky/H)
            kx = torch.arange(W, device=device, dtype=torch.float32)
            ky = torch.arange(H, device=device, dtype=torch.float32)
            lam_x = 2.0 - 2.0 * torch.cos(2.0 * math.pi * kx / W)  # shape (W,)
            lam_y = 2.0 - 2.0 * torch.cos(2.0 * math.pi * ky / H)  # shape (H,)
            lam = lam_y.view(H, 1) + lam_x.view(1, W)              # (H,W), fp32
            self._lam_cache[key] = lam
        return self._lam_cache[key]

    def _radial_highk_boost(self, H, W, device, dtype):
        # frequency indices in "index units" (so Nyquist per axis is 64 on 128x128)
        fy = torch.fft.fftfreq(H, d=1.0).to(dtype=dtype, device=device).view(H,1) * H
        fx = torch.fft.fftfreq(W, d=1.0).to(dtype=dtype, device=device).view(1,W) * W
        r  = torch.sqrt(fy*fy + fx*fx)  # (H,W)
        B = 1.0 + torch.exp(self.log_raw_boost)
        # soft wall that turns on beyond k_cut
        sig = torch.sigmoid(torch.as_tensor(self.eta, dtype=dtype, device=device) * (r - self.k_cut))
        return 1.0 + (B - 1.0) * sig  # (H,W)

    def spectral_log_prior(self, x1):  # x1: (B,H,W) or (B,C,H,W)
        # Sum per-channel if provided
        if x1.dim() == 4:
            return sum(self.spectral_log_prior(x1[:, c]) for c in range(x1.size(1)))

        B, H, W = x1.shape
        D = H * W
        two_pi_log = math.log(2.0 * math.pi)

        # Compute in fp32 (AMP-safe), return in input dtype
        out_dtype = x1.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x = x1.float()                                 # fp32
            X = torch.fft.fft2(x, norm="ortho")            # unitary DFT

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
            power = alpha + beta * lam + gamma * (lam ** 2)                     # (H,W) > 0
            
            # Optional safety if alpha is tiny:
            # power = power.clamp_min(1e-12)
            
            # Quadratic form x^T (αI + βΔ) x  via Parseval (unitary)
            quad = (power * (X.real * X.real + X.imag * X.imag)).sum(dim=(-1, -2))  # (B,)
            
            # Log-normalizer: 0.5 * sum log(power) - 0.5 * D * log(2π)
            const = 0.5 * power.log().sum() - 0.5 * D * two_pi_log                  # scalar
            
            lp = (-0.5 * quad + const).to(out_dtype)                                # (B,)
            """
            return lp

    def sample_x1(self, y: Tensor):
        """Dispatch to the selected posterior family."""
        if self.posterior in ["flow", "flow_Spline", "flow_RealNVP"]:
            return self.sample_x1_flow(y)
        elif self.posterior == "diag_gauss":
            return self.sample_x1_diag_gauss(y)
        else:
            raise ValueError(f"Unsupported posterior {self.posterior}")

    def sample_x1T(self, y: Tensor):
        """Dispatch to the selected posterior family."""
        if self.posterior in ["flow", "flow_Spline", "flow_RealNVP"]:
            return self.sample_x1T_flow(y)
        elif self.posterior == "diag_gauss":
            return self.sample_x1T_diag_gauss(y)
        else:
            raise ValueError(f"Unsupported posterior {self.posterior}")

    # ------------------------------------------------------------------
    def sample_x1_flow(self, y: Tensor):
        """Sample X1 ~ q_φ(X1|Y) via conditional flow; return (x1, logq_x1)."""
        # ---- 1) Normalise observations
        y_scaled = y / self.norm_factor

        # ---- 2) Encode to get context features (reuse your existing heads)
        h = self.encoder(y_scaled)     # expected ~ (B, 2, D) or (B, 2, H, W)

        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            # h: (B, 2, D) where channels are [μ_pred, logσ_pred] (we only use them to build context)
            assert h.dim() == 3 and h.size(2) == self.latent_dim
            ctx = h.flatten(1)
            ctx = self.flow_ctx_proj(ctx)                                # (B, Cctx)
            x1, logq_x1 = self.flow.sample_and_logq(B=h.size(0), D=self.latent_dim, c=ctx, device=h.device)
            assert torch.isfinite(x1).all() and torch.isfinite(logq_x1).all(), "flow produced NaN/Inf"
            return x1, logq_x1

        elif self.dataset in ["Kolmogorov"]:
            # h: (B, 2, H, W) or (B, 2, D) -> reshape to (B,2,H,W)
            if h.dim() == 3:
                B, C, D = h.shape
                HW = int(math.isqrt(D))
                assert HW * HW == D, f"latent_dim {D} not a perfect square for 2D flow"
                h2 = h.view(B, C, HW, HW)
            else:
                h2 = h
            B, C, H, W = h2.shape
            assert C == 2, f"Encoder must return 2 channels (got C={C})"
            c_map = self.ctx_conv(h2)  # (B, ctx_ch, H, W)

            # Sample *map* (B,1,H,W) and squeeze channel -> (B,H,W)
            x1_map, logq_x1 = self.flow.sample_and_logq(
                B=B, spatial_shape=(1, H, W), c_map=c_map, device=h.device, dtype=h.dtype
            )
            x1 = x1_map.squeeze(1)  # (B,H,W)
            return x1, logq_x1
        else:
            raise ValueError(f"Unsupported dataset {self.dataset} for flow posterior")

    # ---------------- New: diagonal-Gaussian posterior path ----------------
    def sample_x1_diag_gauss(self, y: Tensor):
        """
        Uses encoder outputs as μ and logσ for a diagonal Gaussian:
        q(X1|Y) = N( μ(Y), diag(σ^2(Y)) )
        Returns (x1, log q(x1|y)).
        """
        y_scaled = y / self.norm_factor
        h = self.encoder(y_scaled)
        
        two_pi_log = math.log(2.0 * math.pi)
        
        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            # h: (B,2,D)
            assert h.dim() == 3 and h.size(1) == 2, f"Encoder must return (B,2,D); got {tuple(h.shape)}"
            mu, logstd = h[:, 0], h[:, 1]                     # (B,D), (B,D)
            # (optional) clamp for stability; adjust bounds if you like
            logstd = logstd.clamp(min=-7.0, max=5.0)
            eps = torch.randn_like(mu)
            x1 = mu + torch.exp(logstd) * eps                 # reparameterized sample
            # log q = -0.5 * (sum eps^2 + D*log(2π)) - sum logσ
            logq = -0.5 * (eps.pow(2).sum(-1) + mu.size(-1) * two_pi_log) - logstd.sum(-1)
            return x1, logq

        elif self.dataset in ["Kolmogorov"]:
            # h: (B,2,H,W) or (B,2,D)
            if h.dim() == 3:
                B, C, D = h.shape
                HW = int(math.isqrt(D)); assert HW * HW == D, "latent_dim not a perfect square for 2D Gaussian"
                h = h.view(B, C, HW, HW)
            B, C, H, W = h.shape
            assert C == 2, f"Encoder must return 2 channels (got C={C})"
            mu, logstd = h[:, 0], h[:, 1]                     # (B,H,W)
            logstd = logstd.clamp(min=-7.0, max=5.0)
            eps = torch.randn_like(mu)
            x1 = mu + torch.exp(logstd) * eps                 # (B,H,W)
            D = H * W
            # sum over spatial dims:
            logq = -0.5 * (eps.pow(2).sum(dim=(-1, -2)) + D * two_pi_log) - logstd.sum(dim=(-1, -2))
            return x1, logq
        else:
            raise ValueError(f"Unsupported dataset {self.dataset} for diag_gauss posterior")

    def sample_x1T_flow(self, y: torch.Tensor): # for noisy path
        """
        Sample the full state path X_{1:T} ~ q_phi(X_{1:T} | Y_{1:T}) via a conditional flow.
        Returns:
        xs:        (B, T, D) for 1-D (Lorenz96 / Lorenz96_4dim)
                   (B, T, H, W) for 2-D (Kolmogorov)
        logq_x1T: (B,) total log-density under the flow
        """

        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            y_scaled = y / self.norm_factor
            B, T, D = y.shape
        
            # ----- encode observations to context -----
            h = self.encoder(y_scaled)

            ctx = h.flatten(1)
            c_global = self.flow_ctx_proj(ctx)         # (B, _flow_ctx_dim)
            
            # per-step contexts from Transformer (future-aware)
            _, c_steps = self.future_ctx(y_scaled)     # (B, T-1, Cs)

            # sanity guards (helpful while wiring)
            assert c_global.shape[-1] == self._flow_ctx_dim
            assert c_steps.shape[1] == T - 1
            #assert c_steps.shape[-1] == 128
            
            # pass BOTH contexts to the noisy flow
            flat, logq = self.flow.sample_and_logq(
                B=B,
                D=self.latent_dim,                     # the wrapper ignores this, keep = latent_dim
                c=(c_global, c_steps),
                device=y_scaled.device
            )
            
            # reshape flattened states to (B, T, D) if your caller expects that
            xs = flat.view(B, T, self.latent_dim)
            assert torch.isfinite(xs).all()
            return xs, logq
        elif self.dataset in ["Kolmogorov"]:
            assert y.dim() == 4
            y_scaled = y / self.norm_factor
        
            # ----- encode observations to context -----
            h = self.encoder(y_scaled)
            print(f"{h.shape=}")
            # y: (B, T, 128, 128)
            assert self.noisy and self.posterior in ["flow"], "noisy Kolmogorov path expects flow posterior"
            B, T, H, W = y_scaled.shape
            print(f"{H=}, {W=}")
            # 1) contexts
            c_global, c_steps = self._build_future_ctx_maps(y_scaled)   # (B,Cc,H,W), (B,T-1,Cc,H,W)
            
            # 2) sample X1 from flow (map‑valued), conditional on *global* context
            x1_map, logq_x1 = self.flow.sample_and_logq(
                B=B, spatial_shape=(1, W, W), c_map=c_global, device=y_scaled.device, dtype=y_scaled.dtype # not 1, H, W: even if the sample 
            )                                                           # (B,1,H,W), (B,)
            x1 = x1_map.squeeze(1)                                      # (B,H,W)

            # 3) rollout with shock head (future‑aware)
            two_pi_log = math.log(2.0 * math.pi)
            xs_list = [x1]
            logq_eps_sum = torch.zeros(B, device=y_scaled.device, dtype=y_scaled.dtype)

            for t in range(T - 1):
                X_t = xs_list[-1]                                       # (B,H,W)
                time_map = torch.full((B,1,W,W), float(t)/max(T-1,1),
                                      device=y_scaled.device, dtype=y_scaled.dtype)
                # input to shock head: [X_t, c_t, time]
                #print(f"{X_t.unsqueeze(1).shape=}, {c_steps[:, t, :, :, :].shape=}, {time_map.shape=}")
                c_t_map = torch.cat([X_t.unsqueeze(1), c_steps[:, t, :, :, :], time_map], dim=1)  # (B, 1+Cc+1, H, W)
                m_t, _logstd_ignored = self.shock_head_2d(c_t_map)      # (B,1,H,W), _
                # Use trainable log_dyn_noise (S_t) rather than the shock head's fixed log-std.
                ls = self.log_dyn_noise.to(device=m_t.device, dtype=m_t.dtype)
                # broadcast scalar → (B,1,H,W)
                if ls.ndim == 0:
                    logstd_t = ls.view(1, 1, 1, 1).expand_as(m_t)
                else:
                    # per-dimension logstd is not meaningful for map-valued Kolmogorov; fall back to scalar.
                    logstd_t = ls.flatten()[0].view(1, 1, 1, 1).expand_as(m_t)
                # log_dyn_noise is already <= 0; no lower bound is imposed.
                logstd_t = logstd_t.clamp_max(0.0)
                z = torch.randn_like(m_t)
                delta_t = m_t + torch.exp(logstd_t) * z                 # (B,1,H,W)
                # log q(ε_t)
                Dsp = H * W
                logq_eps_t = -0.5 * (z.pow(2).sum(dim=(1,2,3)) + Dsp * two_pi_log) - logstd_t.sum(dim=(1,2,3))
                # physics + shock (match L96 clamp)
                with torch.cuda.amp.autocast(enabled=False):
                    f_t = self.dyn.sample(X_t.float())                              # (B,H,W) deterministic integrate step here
                X_next = torch.clamp(f_t + delta_t.squeeze(1), -20, 20)
                xs_list.append(X_next); logq_eps_sum += logq_eps_t

            xs = torch.stack(xs_list, dim=1)                            # (B,T,H,W)
            logq_total = logq_x1 + logq_eps_sum
            return xs, logq_total
        else:
            raise ValueError(f"Unsupported dataset {self.dataset} for full-path flow posterior")

    def sample_x1T_diag_gauss(self, y: torch.Tensor):
        """
        Sample the full path X_{1:T} under a diagonal-Gaussian initial posterior:
        q(X1|Y) = N(mu(Y), diag(sigma^2(Y))).
        Uses the SAME per-step contexts (c_steps) and shock_head as the noisy flow:
        delta_t ~ N(m_t(X_t, c_steps_t, time_frac), diag(exp(2*logstd_t))).
        Returns:
        xs:       (B, T, D)  for Lorenz96 / Lorenz96_4dim
        logq:     (B,)       total variational log-density log q(X_{1:T}|Y)
        """
        y_scaled = y / self.norm_factor
        B, T = int(y.size(0)), int(y.size(1))
        
        # Encode observations → initial posterior params for X1
        h = self.encoder(y_scaled)
        
        if self.dataset in ["Lorenz96", "Lorenz96_4dim"]:
            # h expected: (B, 2, D) with channels [mu, logstd]
            assert h.dim() == 3 and h.size(1) == 2, f"Encoder must return (B,2,D); got {tuple(h.shape)}"
            D = int(h.size(2))
            mu, logstd = h[:, 0], h[:, 1]            # (B,D), (B,D)
            logstd = logstd.clamp(min=-7.0, max=5.0) # stability
            
            # X1 ~ diag-Gauss(mu, exp(2*logstd))
            two_pi_log = math.log(2.0 * math.pi)
            z0 = torch.randn_like(mu)
            x1 = mu + torch.exp(logstd) * z0
            logq_x1 = -0.5 * (z0.pow(2).sum(-1) + D * two_pi_log) - logstd.sum(-1)  # (B,)
            
            # ---- contexts (match sample_x1T_flow wiring) ----
            # c_global is *not* used here, but we compute+guard to keep the same wiring & checks.
            ctx = h.flatten(1)                        # (B, 2*D)
            c_global = self.flow_ctx_proj(ctx)        # (B, _flow_ctx_dim)
            _, c_steps = self.future_ctx(y_scaled)    # (B, T-1, Cs)
            
            assert c_steps.shape[1] == max(T - 1, 0), f"c_steps time len={c_steps.shape[1]} vs T-1={T-1}"
            #assert c_steps.shape[-1] == 128, f"expected Cs=128, got {c_steps.shape[-1]}"
            assert hasattr(self, "flow") and hasattr(self.flow, "shock_head") and hasattr(self.flow, "f_theta"), \
                "self.flow.shock_head / self.flow.f_theta required for noisy rollout"

            # ---- rollout with shocks from the SAME head as the flow ----
            xs_list = [x1]
            logq_eps_sum = torch.zeros(B, device=y.device, dtype=y.dtype)

            for t in range(T - 1):
                X_t = xs_list[-1]                     # (B,D)
                time_frac = torch.full(
                    (B, 1), float(t) / max(T - 1, 1),
                    device=y.device, dtype=y.dtype
                )
                # shock context = [state, step-context, time]
                c_t = torch.cat([X_t, c_steps[:, t, :], time_frac], dim=-1)  # (B, D+Cs+1)
                
                # Per-step diag-Gaussian for delta_t.
                # Mean m_t comes from the shock head; covariance S_t is controlled by PRSmoother.log_dyn_noise.
                m_t, _logstd_ignored = self.flow.shock_head(c_t)    # (B,D), _
                ls = self.log_dyn_noise.to(device=m_t.device, dtype=m_t.dtype)
                if ls.ndim == 0:
                    logstd_t = ls.expand_as(m_t)
                else:
                    logstd_t = ls.view(1, -1).expand_as(m_t)
                # log_dyn_noise is already <= 0; no lower bound is imposed.
                logstd_t = logstd_t.clamp_max(0.0)
                z = torch.randn_like(m_t)
                delta_t = m_t + torch.exp(logstd_t) * z
                logq_eps_t = -0.5 * (z.pow(2).sum(-1) + D * two_pi_log) - logstd_t.sum(-1)  # (B,)
                
                # Physics step + shock (match NoisyGaussianB1 clamp)
                X_next = torch.clamp(self.flow.f_theta(X_t) + delta_t, -20, 20)
                assert torch.isfinite(X_next).all(), "NaN/Inf encountered in rollout"
                
                xs_list.append(X_next)
                logq_eps_sum += logq_eps_t
                
            xs = torch.stack(xs_list, dim=1)          # (B,T,D)
            logq_total = logq_x1 + logq_eps_sum       # (B,)
            return xs, logq_total
        elif self.dataset in ["Kolmogorov"]:
            assert self.noisy and (self.posterior == "diag_gauss" or self.posterior == "flow"), \
                "noisy Kolmogorov path expects diag_gauss or flow posterior"
            B, T, H, W = y_scaled.shape
            two_pi_log = math.log(2.0 * math.pi)
            
            # 1) X1 ~ diag Gaussian from encoder (μ, logσ)
            h = self.encoder(y_scaled)                                   # (B,2,H,W)
            if h.dim() == 3:
                B2, C2, D2 = h.shape
                HW = int(math.isqrt(D2)); assert HW * HW == D2
                h = h.view(B2, C2, HW, HW)
            mu1, logstd1 = h[:, 0], h[:, 1]                              # (B,H,W)
            logstd1 = logstd1.clamp(min=-7.0, max=5.0)
            z0 = torch.randn_like(mu1)
            x1 = mu1 + torch.exp(logstd1) * z0                           # (B,H,W)
            Dsp = H * W
            logq_x1 = -0.5 * (z0.pow(2).sum(dim=(1,2)) + Dsp * two_pi_log) - logstd1.sum(dim=(1,2))

            # 2) contexts: future‑only maps per step
            c_global, c_steps = self._build_future_ctx_maps(y_scaled)    # (B,Cc,H,W), (B,T-1,Cc,H,W)
            
            # 3) rollout with shock head (same as flow path)
            xs_list = [x1]
            logq_eps_sum = torch.zeros(B, device=y_scaled.device, dtype=y_scaled.dtype)
            
            for t in range(T - 1):
                X_t = xs_list[-1]
                time_map = torch.full((B,1,W,W), float(t)/max(T-1,1),
                                      device=y_scaled.device, dtype=y_scaled.dtype)
                c_t_map = torch.cat([X_t.unsqueeze(1), c_steps[:, t, :, :, :], time_map], dim=1)
                m_t, _logstd_ignored = self.shock_head_2d(c_t_map)       # (B,1,H,W), _
                ls = self.log_dyn_noise.to(device=m_t.device, dtype=m_t.dtype)
                if ls.ndim == 0:
                    logstd_t = ls.view(1, 1, 1, 1).expand_as(m_t)
                else:
                    logstd_t = ls.flatten()[0].view(1, 1, 1, 1).expand_as(m_t)
                # log_dyn_noise is already <= 0; no lower bound is imposed.
                logstd_t = logstd_t.clamp_max(0.0)
                z = torch.randn_like(m_t)
                delta_t = m_t + torch.exp(logstd_t) * z
                logq_eps_t = -0.5 * (z.pow(2).sum(dim=(1,2,3)) + Dsp * two_pi_log) - logstd_t.sum(dim=(1,2,3))
                with torch.cuda.amp.autocast(enabled=False):
                    f_t = self.dyn.sample(X_t.float())
                X_next = torch.clamp(f_t + delta_t.squeeze(1), -20, 20)
                xs_list.append(X_next)
                logq_eps_sum += logq_eps_t
            xs = torch.stack(xs_list, dim=1)                              # (B,T,H,W)
            return xs, (logq_x1 + logq_eps_sum)
        else:
            raise ValueError(f"Unsupported dataset {self.dataset} for full-path diag_gauss posterior")


    # ------------------------------------------------------------------
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
    
    def obs_log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        """
        Log p(y|x) for diagonal Gaussian with variance σ² = exp(self.log_r).
        Supports:
          - L96: y, x shapes (B,T,D)  (and sparse uses ::4 on last dim)
          - 2D:  y, x shapes (B,T,H,W); sparse downsamples both axes to y's H,W
        """
        two_pi = torch.tensor(2.0 * math.pi, dtype=y.dtype, device=y.device)
        var = torch.exp(self.log_r)  # scalar σ²

        # L96 (vector) case
        if x.dim() == 3 and y.dim() == 3:
            if self.observation == "full_linear":
                diff = y - x
                #diff = y - (x+self.bias)
            elif self.observation == "full_nonlinear":
                #x_to_4_clamped = torch.clamp((x+self.bias)**4, 0, 10)
                x_to_2_clamped = torch.clamp(x**2, 0, 5)
                diff = y - x_to_2_clamped
            elif self.observation == "full_pm3":
                diff = y - torch.clamp(x+self.bias, -3.0, 3.0)
            elif self.observation == "square":
                diff = y - (x*x+self.bias)
            elif self.observation == "biased_abs":
                diff = y - (abs(x-2)+self.bias)
            elif self.observation == "sparse":
                diff = y - x[..., ::4]+self.bias
            elif self.observation == "sparse_pm3":
                diff = y - torch.clamp(x[..., ::4]+self.bias, -3.0, 3.0)
            else:
                raise ValueError(f"Unknown observation mode {self.observation}")
            return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(-1)

        # 2D field case
        if x.dim() == 4 and y.dim() == 4:
            # x: (B,T,Hx,Wx), y: (B,T,Hy,Wy)
            Hx, Wx = x.shape[-2], x.shape[-1]
            Hy, Wy = y.shape[-2], y.shape[-1]
            if self.observation in ["full", "full_pm3"]:
                x_for_y = x+self.bias
                if self.observation.endswith("pm3"):
                    x_for_y = torch.clamp(x_for_y, -3.0, 3.0)
                diff = y - x_for_y
                return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(dim=(-1, -2))
            elif self.observation in ["half"]:
                x_for_y = x[:, :, :64, :]+self.bias
                if self.observation.endswith("pm3"):
                    x_for_y = torch.clamp(x_for_y, -3.0, 3.0)
                diff = y - x_for_y
                return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(dim=(-1, -2))
            elif self.observation in ["coarse"]:
                x_for_y = F.avg_pool2d(x, kernel_size=8, stride=8)+self.bias
                diff = y - x_for_y
                return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(dim=(-1, -2))
            elif self.observation in ["sparse_pm3"]:
                """
                assert Hx % Hy == 0 and Wx % Wy == 0, "x and y must be integer-factor related"
                sh, sw = Hx // Hy, Wx // Wy
                x_ds = x[..., 3::sh, 3::sw]  # (B,T,Hy,Wy)
                if self.observation.endswith("pm3"):
                    x_ds = torch.clamp(x_ds, -3.0, 3.0)
                diff = y - x_ds
                return -0.5 * ((diff ** 2) / var + torch.log(two_pi * var)).sum(dim=(-1, -2))
                """
            elif self.observation in ["sparse", "mixed", "mixed_pm3"]:
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

        raise ValueError(f"Unexpected x,y shapes: x{tuple(x.shape)} y{tuple(y.shape)}")

    # ------------------------------------------------------------------
    def forward(self, y: Tensor) -> Tensor:
        if self.dataset in ["Lorenz96"]:
            B, T, d_obs = y.shape
            if self.observation in ["full_linear", "full_pm3", "full_nonlinear"]:
                assert T == 50 and d_obs == self.latent_dim, "Unexpected input shape"
            elif self.observation in ["sparse", "sparse_pm3"]:
                assert T == 50 and d_obs == int(self.latent_dim)//4, "Unexpected input shape"
        elif self.dataset in ["Kolmogorov"]:
            B, T, d_obs, d_obs = y.shape
            if self.observation in ["full", "full_pm3", "sparse"]:
                assert T == 10 and d_obs*d_obs == self.latent_dim, "Unexpected input shape"
            elif self.observation in ["half", "coarse", "mixed"]:
                assert T == 10, "Unexpected input shape"

        if self.noisy:
            #print("noisy path")
            # ===========================
            #      NOISY DYNAMICS PATH
            # ===========================
            # q(X_{1:T} | Y_{1:T}) from the flow; evaluate Gaussian dynamics prior
            T = y.size(1)
            
            # 1) Sample full path and its log q
            xs, logq_x1T = self.sample_x1T(y)#self.sample_x1T_flow(y)  # xs: (B,T,...) ; logq: (B,)
            assert xs.shape[1] == T
            
            # Safety & dtype for FFT
            if not torch.isfinite(xs).all():
                return 0, xs, 0, 0, 0
            with torch.cuda.amp.autocast(enabled=False):
                xs = xs.float()

            # 2) Observation term
            lp_obs = self.obs_log_prob(y.to(xs.dtype), xs)  # (B,T) or (B,) then sum over time below

            # 3) Prior terms: p(x1) and ∑_{t=1}^{T-1} p(x_{t+1} | x_t)
            log_2pi = math.log(2.0 * math.pi)
            
            # Prior on x1
            x1 = xs[:, 0]
            if x1.dim() == 3:  # (B,H,W)
                lp_prior_x1 = self.spectral_log_prior(x1)  # (B,)
                D_step = (x1.numel() // x1.size(0))
            else:
                x1_flat = x1.view(x1.size(0), -1)
                D_step = x1_flat.size(-1)
                lp_prior_x1 = -0.5 * (
                    (x1_flat.pow(2).sum(-1) / self.sigma0_sq)
                    + D_step * (log_2pi + math.log(self.sigma0_sq))
                )
                
            # Dynamics prior: x_{t+1} ~ N(f(x_t), Q) where Q is the *process* noise.
            # IMPORTANT: Q is NOT controlled by `log_dyn_noise` (which is for S_t only).
            log_q_std = self._log_process_noise_std
            if not isinstance(log_q_std, torch.Tensor):
                log_q_std = torch.as_tensor(log_q_std, device=xs.device, dtype=xs.dtype)
            else:
                log_q_std = log_q_std.to(device=xs.device, dtype=xs.dtype)
                
            per_dim_Q = (log_q_std.ndim > 0)
            if per_dim_Q:
                # vector of variances shaped to (1,1,D)
                q_var_vec = torch.exp(2.0 * log_q_std).view(1, 1, -1)
                log_norm_vec = (torch.log(q_var_vec) + log_2pi).sum(-1)  # (1,1)
            else:
                q_var = torch.exp(2.0 * log_q_std)
                log_norm_scalar = D_step * (log_2pi + torch.log(q_var))

            lp_dyn_terms = []
            with torch.cuda.amp.autocast(enabled=False):
                for t in range(T - 1):
                    x_t   = xs[:, t]
                    x_tp1 = xs[:, t + 1]
                    f_t   = self.dyn.sample(x_t)  # deterministic step f(x_t)
                    
                    delta = (x_tp1 - f_t).view(x_t.size(0), -1)  # (B,D_step)
                    if per_dim_Q:
                        term = -0.5 * ((delta**2) / q_var_vec.view(1, 1, -1)).sum(-1).squeeze(1)
                        term = term - 0.5 * log_norm_vec.squeeze(1).expand_as(term)
                    else:
                        term = -0.5 * (delta.pow(2).sum(-1) / q_var) - 0.5 * log_norm_scalar
                    lp_dyn_terms.append(term)

            lp_dyn = torch.stack(lp_dyn_terms, dim=1).sum(1)   # (B,)
            lp_prior_total = lp_prior_x1 + lp_dyn               # (B,)

            # 4) Posterior entropy term (only log q of the full path)
            lq_total = logq_x1T.to(xs.dtype)                    # (B,)

            # 5) ELBO
            assert torch.isfinite(lp_obs).all() and torch.isfinite(lp_prior_total).all() and torch.isfinite(lq_total).all()
            elbo1 = lp_prior_total.mean()       # prior(x1) + sum_t prior dynamics
            elbo2 = lp_obs.sum(1).mean()        # sum_t log p(y_t|x_t)
            elbo3 = lq_total.mean()             # log q(x_{1:T}|y)
            elbo = elbo1 + elbo2 - elbo3
            return -elbo, xs, elbo1.detach(), elbo2.detach(), elbo3.detach()
        else:
            # ---- posterior sample and log q via conditional flow ----
            x1, logq_x1 = self.sample_x1(y)   # may be bf16 due to AMP
            # ---- dynamics rollout in float32 (FFT-safe) ----
            T = y.size(1)
            with torch.cuda.amp.autocast(enabled=False):
                x_cur = x1.float()
                xs = [x_cur]
                for _ in range(1, T):
                    x_cur = self.dyn.sample(x_cur)     # Kolmogorov expects (B,H,W) float32
                    xs.append(x_cur)
                xs = torch.stack(xs, dim=1)            # xs: float32

            if not torch.isfinite(xs).all():
                return 0, xs, 0, 0, 0
            assert torch.isfinite(xs).all(), "dynamics produced NaN/Inf"
            # ---- observation term: compute in the rollout dtype ----
            lp_obs = self.obs_log_prob(y.to(xs.dtype), xs)
            log_2pi = math.log(2.0 * math.pi)
            
            x1_flat = x1.view(x1.size(0), -1).to(xs.dtype)
            D_eff = x1_flat.size(-1)
            if x1.dim() == 3:     # (B,H,W)
                lp_prior = self.spectral_log_prior(x1)
            else:
                lp_prior = -0.5 * (
                    (x1_flat.pow(2).sum(-1) / self.sigma0_sq)
                    + D_eff * (log_2pi + math.log(self.sigma0_sq))
                )
        
            lq = logq_x1.to(xs.dtype)
            assert torch.isfinite(lp_obs).all(), "obs logprob NaN/Inf"
            assert torch.isfinite(lp_prior).all(), "prior NaN/Inf"
            elbo1 = lp_prior.mean()
            elbo2 = lp_obs.sum(1).mean()
            elbo3 = lq.mean()
            elbo = elbo1 + elbo2 - elbo3
            return -elbo, xs, elbo1.detach(), elbo2.detach(), elbo3.detach()
