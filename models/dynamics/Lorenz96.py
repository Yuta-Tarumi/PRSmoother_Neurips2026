import torch                      # main PyTorch package
import torch.nn as nn             # neural-network helpers
from torch import Tensor          # for type hints

def l96_rhs(x: Tensor, F: Tensor) -> Tensor:
    x_ip1 = torch.roll(x, shifts=-1, dims=-1)
    x_im2 = torch.roll(x, shifts= 2, dims=-1)
    x_im1 = torch.roll(x, shifts= 1, dims=-1)
    return (x_ip1 - x_im2) * x_im1 - x + F

def rk4_step_l96(x: Tensor, dt: float, F: Tensor) -> Tensor:
    k1 = l96_rhs(x,             F)
    k2 = l96_rhs(x + 0.5*dt*k1, F)
    k3 = l96_rhs(x + 0.5*dt*k2, F)
    k4 = l96_rhs(x + dt*k3,     F)
    return x + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

def integrate_l96(
    x: Tensor,
    F: Tensor,
    obs_dt: float,
    internal_dt: float,
) -> Tensor:
    """Propagate one observation interval by k = obs_dt / internal_dt RK4 steps."""
    k = int(round(obs_dt / internal_dt))
    x_dtype = x.dtype
    with torch.cuda.amp.autocast(enabled=False):
        x32 = x.float()
        F32 = F.to(dtype=torch.float32)
        for _ in range(k):
            x32 = rk4_step_l96(x32, dt=internal_dt, F=F32)
    return x32.to(x_dtype)

class L96Dynamics(nn.Module):
    def __init__(self, init_F: float = 8.0, init_log_sigma: float = -4.5, train_F: bool = True, train_Q: bool = True,
                 obs_dt: float = 0.03, internal_dt: float = 0.003):
        super().__init__()
        if train_F:
            self.F = nn.Parameter(torch.tensor(init_F))          # scalar
        else:
            self.F = nn.Parameter(torch.tensor(init_F), requires_grad=False)          # scalar
        if train_Q:
            self.log_sigma  = nn.Parameter(torch.tensor(init_log_sigma))  # scalar
        else:
            self.log_sigma  = nn.Parameter(torch.tensor(init_log_sigma), requires_grad=False)  # scalar
        self.obs_dt     = obs_dt
        self.internal_dt = internal_dt

    # ---- stochastic transition ------------------------------------------------
    def sample(self, x_prev: Tensor) -> Tensor:
        mu_t  = torch.clamp(integrate_l96(x_prev, self.F, self.obs_dt, self.internal_dt), -20, 20)
        sigma = torch.exp(self.log_sigma)                              # scalar
        eps   = torch.randn_like(x_prev)
        return mu_t + sigma * eps                                      # Q = σ² I

    # ---- log-probability for ELBO --------------------------------------------
    def log_prob(self, x_t: Tensor, x_prev: Tensor) -> Tensor:
        mu_t = integrate_l96(x_prev, self.F, self.obs_dt, self.internal_dt)
        var  = torch.exp(2.0 * self.log_sigma)                         # σ²
        return -0.5 * (
            ((x_t - mu_t)**2) / var
            + 2.0 * self.log_sigma
            + torch.log(torch.tensor(2.0 * torch.pi), device="cuda")
        ).sum(-1)                                                      # (batch,)

    def forward(self, x_prev: Tensor, n_obs: int = 1) -> Tensor:
        return torch.clamp(integrate_l96(x_prev, self.F, self.obs_dt, self.internal_dt), -20, 20)

