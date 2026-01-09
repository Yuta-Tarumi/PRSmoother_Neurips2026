# --- add/replace your class with this version --------------------------------                                                                                                                                                     
from __future__ import annotations
import math
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor

_TWO_PI = 2.0 * math.pi
_LOG2PI = math.log(_TWO_PI)

class KolmogorovDynamics(nn.Module):
    def __init__(
        self,
        *,
        N: int = 128,
        L: float = _TWO_PI,
        k_force: int = 4,
        internal_dt: float = 5.0e-3,
        obs_dt: float = 5.0e-2,
        gamma: float = 0.1,
        init_logRe: float = math.log(100.0),
        init_F0: float = 0.1,
        init_log_sigma: float = math.log(1e-100),
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.N, self.L, self.dt = N, L, float(internal_dt)
        self.obs_dt = float(obs_dt)
        self.kf, self.gamma = int(k_force), float(gamma)

        steps_per_obs = self.obs_dt / self.dt
        if abs(round(steps_per_obs) - steps_per_obs) > 1e-9:
            raise ValueError("obs_dt must be an integer multiple of internal_dt")
        self.steps_per_obs = int(round(steps_per_obs))

        # learnable / physical params                                                                                                                                                                                               
        self.logRe = nn.Parameter(torch.tensor(init_logRe, dtype=dtype, device=device))
        self.F0 = nn.Parameter(torch.tensor(init_F0, dtype=dtype, device=device))
        self.log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=dtype, device=device))
        #self.logRe = torch.log(torch.tensor(1000, device=device))
        #self.F0 = torch.tensor(1.0, device=device)
        #self.log_sigma = torch.tensor(-10.0, device=device)

        # spectral operators (centers)
        
        k = (_TWO_PI / L) * torch.fft.fftfreq(N, d=1.0 / N, device=device, dtype=dtype)
        kx, ky = torch.meshgrid(k, k, indexing="ij")
        k2 = (kx**2 + ky**2).clamp_min(1e-14)
        
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("k2", k2)
        self.register_buffer("k4", k2**2)
        self.register_buffer("laplacian", -k2)
        #self.register_buffer("k2_mac", kx**2 + ky**2)

        k2_safe = k2.clone()
        k2_safe[0, 0] = 1.0
        self.register_buffer("k2_safe", k2_safe)

        # 2/3 dealias

        k_cut = (2.0 / 3.0) * k.abs().max()
        dealias_mask = (kx.abs() <= k_cut) & (ky.abs() <= k_cut)
        self.register_buffer("dealias_mask", dealias_mask)

        rho = ((kx.abs() / k.abs().max())**2 + (ky.abs() / k.abs().max())**2).clamp(0, 1)
        spec_filter = torch.exp(-36.0 * (rho**18))
        spec_filter[0, 0] = 1.0
        self.register_buffer("spec_filter", spec_filter)

        # periodic grid (endpoint=False)
        dx = L / N
        x = torch.arange(N, device=device, dtype=dtype) * dx
        y = x
        forcing_template = -self.kf * torch.cos(self.kf * y).view(1, N, 1)  # vorticity curl of Fx                                                                                                                                  
        self.register_buffer("forcing_template", forcing_template)
        F_phys = self.forcing_template.squeeze(-1).expand(self.N, self.N).contiguous()#(-self.kf * torch.cos(self.kf * y)).repeat(N, 1)           # (N, N)
        F_hat  = torch.fft.fft2(F_phys, norm="forward")
        self.register_buffer("F_hat", F_hat)

        self._Nh_prev = None

        kmax = float(k_cut)
        self.nu4 =  0.1 / (self.dt * (kmax**4))

    # ---------------- spectral-vorticity RHS (RK4 path) ----------------                                                                                                                                                           
    def _adv_flux(self, w, u, v):
        qx = u * w
        qy = v * w
        # mask the flux spectra before taking derivatives                                                                                                                                                                           
        qx_hat = torch.fft.fft2(qx, norm="forward") * self.dealias_mask
        qy_hat = torch.fft.fft2(qy, norm="forward") * self.dealias_mask
        dqx_dx = torch.fft.ifft2(1j * self.kx * qx_hat, norm="forward").real
        dqy_dy = torch.fft.ifft2(1j * self.ky * qy_hat, norm="forward").real
        adv = dqx_dx + dqy_dy
        # final safety mask                                                                                                                                                                                                         
        adv_hat = torch.fft.fft2(adv, norm="forward") * self.dealias_mask
        return torch.fft.ifft2(adv_hat, norm="forward").real

    def _rhs(self, w):
        if w.ndim == 2: w = w.unsqueeze(0)
        u, v = self._velocity_from_vorticity(w)
        adv = self._adv_flux(w, u, v)                      # <-- use flux form                                                                                                                                                      

        Re = self.logRe.exp().clamp_min(1e-6); nu = 1.0 / Re
        lap_w = nu * self._laplacian(w)
        forcing = self.F0 * self.forcing_template
        hyp = -self.nu4 * self._bi_laplacian(w)            # your ν4 term                                                                                                                                                           

        return -adv + lap_w + forcing - self.gamma * w + hyp

    # ---------------- public integrate API ----------------                                                                                                                                                                        
    def forward(self, w: Tensor, n_obs: int = 1) -> Tensor:
        return self.integrate(w, n_obs=n_obs)

    def integrate(
        self,
        w: Tensor,
        *,
        n_obs: int = 1,
        checkpoint_every: Optional[int] = None,
    ) -> Tensor:
        return self._integrate_torch(w, n_obs, checkpoint_every)

    # --------- RK4 (spectral-vorticity) ----------                                                                                                                                                                                 
    def _integrate_torch(self, w: Tensor, n_obs: int, k: Optional[int]) -> Tensor:
        self._Nh_prev = None
        total_steps = int(n_obs) * self.steps_per_obs
        if k and k > 0 and self.training:
            for start in range(0, total_steps, k):
                n_inner = min(k, total_steps - start)
                w = torch.utils.checkpoint.checkpoint(
                    self._rk_block, w, n_inner, use_reentrant=False
                )
        else:
            for _ in range(total_steps):
                w = self._rk4_step(w)
                #w = self._safe_step(w, self._rk2_step)
                #w = self._safe_step(w, self._step_cnab2)

        self._Nh_prev = None
        return w

    def _rk_block(self, w: Tensor, n_inner: int) -> Tensor:
        for _ in range(int(n_inner)):
            w = self._rk4_step(w)
            #w = self._safe_step(w, self._rk2_step)
            #w = self._safe_step(w, self._step_cnab2)
        return w

    def _rk4_step(self, w: Tensor) -> Tensor:
        k1 = self._rhs(w)
        k2 = self._rhs(w + 0.5 * self.dt * k1)
        k3 = self._rhs(w + 0.5 * self.dt * k2)
        k4 = self._rhs(w + self.dt * k3)
        return w + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _safe_step(self, w, step_fn):
        w_next = step_fn(w)
        if not torch.isfinite(w_next).all():
            raise FloatingPointError("NaN/Inf in state after integrator step")
        return w_next

    def _rk2_step(self, w: Tensor) -> Tensor:
        k1 = self._rhs(w)
        w_tilde = w + self.dt * k1
        k2 = self._rhs(w_tilde)
        return w + 0.5 * self.dt * (k1 + k2)

    def _split_rhs(self, w):
        """
        Returns (N_phys, L_phys) such that:
        self._rhs(w) = L_phys + N_phys
        with L_phys = nu*Δw - gamma*w - nu4*∇⁴w,
         N_phys = -adv + F0*forcing_template
        Shapes follow your RK path: w may be (N,N) or (1,N,N); returns (N,N).
        """
        if w.ndim == 3:  # (1,N,N) -> (N,N)
            w2 = w[0]
        else:
            w2 = w
            
        Re = self.logRe.exp().clamp_min(1e-6)
        nu = 1.0 / Re
        
        lap_w   = nu   * self._laplacian(w2)
        bilap_w = self._bi_laplacian(w2)
        lin_phys = lap_w - self.gamma * w2 - self.nu4 * bilap_w
        
        rhs_phys = self._rhs(w2)     # uses your flux form + forcing exactly
        N_phys   = rhs_phys - lin_phys
        return N_phys, lin_phys

    def _step_cnab2(self, w):
        # Linear eigenvalues in Fourier: Δ -> -k^2, ∇⁴ -> +k^4
        Re = self.logRe.exp().clamp_min(1e-6); nu = 1.0 / Re
        lam = -(nu * self.k2 + self.gamma + self.nu4 * self.k4)
        
        A = 1.0 + 0.5 * self.dt * lam
        B = 1.0 - 0.5 * self.dt * lam
        B_inv = 1.0 / B
        
        # Split your RK RHS into explicit N and implicit L (both in physical space)
        N_phys, _ = self._split_rhs(w)                 # N = -adv + F0*forcing (physical)
        w_hat     = torch.fft.fft2(w,      norm="forward")
        Nh        = torch.fft.fft2(N_phys, norm="forward")  # explicit term in Fourier
        
        # AB2 for explicit term; CN for linear part
        incr = self.dt * Nh if self._Nh_prev is None else self.dt * (1.5 * Nh - 0.5 * self._Nh_prev)
        w_hat_next = (A * w_hat + incr) * B_inv * self.spec_filter 
        self._Nh_prev = Nh

        return torch.fft.ifft2(w_hat_next, norm="forward").real
    
    # --------- MAC (staggered, semi-implicit) ----------                                                                                                                                                                           
    # helpers: center<->face, divergence, gradient (all periodic)                                                                                                                                                                   
    def _cx(self, u):  # face-x -> center                                                                                                                                                                                           
        return 0.5 * (u + torch.roll(u, shifts=-1, dims=-2))
    def _cy(self, v):  # face-y -> center                                                                                                                                                                                           
        return 0.5 * (v + torch.roll(v, shifts=-1, dims=-1))
    def _fx(self, c):  # center -> face-x                                                                                                                                                                                           
        return 0.5 * (c + torch.roll(c, shifts=+1, dims=-2))
    def _fy(self, c):  # center -> face-y                                                                                                                                                                                           
        return 0.5 * (c + torch.roll(c, shifts=+1, dims=-1))

    def _div_f(self, u, v):  # faces -> center                                                                                                                                                                                      
        return (u - torch.roll(u, +1, -2))/self.dx + (v - torch.roll(v, +1, -1))/self.dx

    def _grad_c2f(self, phi):  # centers -> faces  (USE FORWARD DIFFS)                                                                                                                                                              
        gx = (torch.roll(phi, -1, -2) - phi) / self.dx   # ∂xφ at x-faces                                                                                                                                                           
        gy = (torch.roll(phi, -1, -1) - phi) / self.dx   # ∂yφ at y-faces                                                                                                                                                           
        return gx, gy

    def _helmholtz_face(self, f, nu: Tensor):
        # Solve (I - ν dt Δ) u = f   →   û = f̂ / (1 + ν dt λ_fd)                                                                                                                                                                    
        with torch.cuda.amp.autocast(enabled=False):
            f32 = f.float()
            fh = torch.fft.fft2(f32, norm="forward")
            denom = 1.0 + nu.float() * float(self.dt) * self.lam_fd.float()  # (N,N)                                                                                                                                                
            uh = fh / denom
            out = torch.fft.ifft2(uh, norm="forward").real
        return out.to(f.dtype)

    def _poisson_center(self, rhs):
        # Solve ∇² φ = rhs with FD Laplacian eigenvalue; avoid DC divide.                                                                                                                                                           
        with torch.cuda.amp.autocast(enabled=False):
            rhs_hat = torch.fft.fft2(rhs.float(), norm="forward")
            lam = self.lam_fd.float()
            denom = lam.clone()
            denom[0, 0] = 1.0                # safe divide                                                                                                                                                                          
            phi_hat = rhs_hat / denom
            phi_hat[..., 0, 0] = 0.0         # exact DC zero                                                                                                                                                                        
            phi = torch.fft.ifft2(phi_hat, norm="forward").real
        return phi.to(rhs.dtype)

    def _advect_faces(self, u, v):
        # u: x-faces (B,Nx,Ny); v: y-faces (B,Nx,Ny)                                                                                                                                                                                
        dx = self.dx

        # Collocate cross velocities:                                                                                                                                                                                               
        # v at x-face centers (average in x), u at y-face centers (average in y)                                                                                                                                                    
        v_at_ux = 0.5 * (v + torch.roll(v, -1, dims=-2))   # align to x-face (i+1/2,j)                                                                                                                                              
        u_at_vy = 0.5 * (u + torch.roll(u, -1, dims=-1))   # align to y-face (i,j+1/2)                                                                                                                                              

        # Fluxes at faces                                                                                                                                                                                                           
        Fxx = 0.5 * (u * u)           # (u^2)/2 at x-faces                                                                                                                                                                          
        Fxy = u * v_at_ux             # uv at x-faces                                                                                                                                                                               
        Gyx = v * u_at_vy             # uv at y-faces                                                                                                                                                                               
        Gyy = 0.5 * (v * v)           # (v^2)/2 at y-faces                                                                                                                                                                          

        # Backward differences (faces -> faces)                                                                                                                                                                                     
        dFxx_dx = (Fxx - torch.roll(Fxx, +1, dims=-2)) / dx
        dFxy_dy = (Fxy - torch.roll(Fxy, +1, dims=-1)) / dx
        dGyx_dx = (Gyx - torch.roll(Gyx, +1, dims=-2)) / dx
        dGyy_dy = (Gyy - torch.roll(Gyy, +1, dims=-1)) / dx

        Nx = dFxx_dx + dFxy_dy
        Ny = dGyx_dx + dGyy_dy
        return Nx, Ny

    def _integrate_mac(self, w: Tensor, n_obs: int) -> Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            added_batch = False
            w32 = w.float()
            if w32.ndim == 2:
                w32 = w32.unsqueeze(0)
                added_batch = True

            # vorticity -> velocities @ centers -> faces                                                                                                                                                                            
            u_c, v_c = self._velocity_from_vorticity(w32)
            u = self._fx(u_c); v = self._fy(v_c)

            assert torch.isfinite(u).all() and torch.isfinite(v).all(), "NaN/Inf: initial faces"

            # initial filter (helps startup)                                                                                                                                                                                        
            u = self._filter_faces(u); v = self._filter_faces(v)
            assert torch.isfinite(u).all() and torch.isfinite(v).all(), "NaN/Inf: after initial filter"

            Re = self.logRe.detach().exp().clamp_min(1e-6).float()
            nu = (1.0 / Re)
            total_steps = int(n_obs) * self.steps_per_obs

            for step in range(total_steps):
                # alpha -> faces                                                                                                                                                                                                    
                a_c, _, _ = self._alpha_and_grads()
                a_x = self._fx(a_c.float()); a_y = self._fy(a_c.float())
                assert torch.isfinite(a_x).all() and torch.isfinite(a_y).all(), f"NaN/Inf: alpha faces (step {step})"

                # forcing on x-faces                                                                                                                                                                                                
                Fx_line = (self.F0.float() * torch.sin(self.kf * self.y_line.float()))  # (N,)                                                                                                                                      
                Fx = Fx_line.view(1, 1, -1).expand_as(u)
                Fy = torch.zeros_like(v)
                assert torch.isfinite(Fx).all() and torch.isfinite(Fy).all(), f"NaN/Inf: forcing (step {step})"

                # filter before advection                                                                                                                                                                                           
                u_f = self._filter_faces(u); v_f = self._filter_faces(v)
                assert torch.isfinite(u_f).all() and torch.isfinite(v_f).all(), f"NaN/Inf: pre-adv filter (step {step})"

                # conservative advection                                                                                                                                                                                            
                Nx, Ny = self._advect_faces(u_f, v_f)
                if not (torch.isfinite(Nx).all() and torch.isfinite(Ny).all()):
                    # Print magnitudes to help debugging                                                                                                                                                                            
                    print(f"[dbg] step {step} | max|u|={float(u.abs().max()):.3e} "
                          f"max|v|={float(v.abs().max()):.3e} "
                          f"max|u_f|={float(u_f.abs().max()):.3e} max|v_f|={float(v_f.abs().max()):.3e} "
                          f"max|Nx|={float(torch.nan_to_num(Nx).abs().max()):.3e} "
                          f"max|Ny|={float(torch.nan_to_num(Ny).abs().max()):.3e} "
                          f"max|Fx|={float(Fx.abs().max()):.3e}")
                    raise RuntimeError(f"NaN/Inf in advection Nx/Ny at step {step}")

                # CFL guardrail for explicit RHS                                                                                                                                                                                    
                with torch.no_grad():
                    umax = torch.maximum(u.abs().amax(dim=(-1, -2), keepdim=True),
                                         v.abs().amax(dim=(-1, -2), keepdim=True)).max()
                scale = float(min(1.0, 0.6 * self.dx / (umax.item() * self.dt + 1e-12)))

                rhs_u = u + self.dt * scale * (-Nx + Fx - self.gamma * u - a_x * u)
                rhs_v = v + self.dt * scale * (-Ny + Fy - self.gamma * v - a_y * v)
                if not (torch.isfinite(rhs_u).all() and torch.isfinite(rhs_v).all()):
                    # Diagnose which term made RHS blow up                                                                                                                                                                          
                    terms = {
                        "u": u, "v": v,
                        "-Nx": -Nx, "-Ny": -Ny,
                        "Fx": Fx, "Fy": Fy,
                        "-γu": -self.gamma * u, "-γv": -self.gamma * v,
                        "-αx u": -a_x * u, "-αy v": -a_y * v,
                    }
                    for name, ten in terms.items():
                        tenf = torch.nan_to_num(ten, nan=0.0, posinf=1e30, neginf=-1e30)
                        print(f"[dbg] step {step} term {name:>6} | max|·|={float(tenf.abs().max()):.3e} "
                              f"finite={bool(torch.isfinite(ten).all())}")
                    print(f"[dbg] step {step} scale={scale:.3e}, dt={self.dt:.3e}, dx={self.dx:.3e}, umax={umax.item():.3e}")
                    raise RuntimeError(f"NaN/Inf in explicit RHS at step {step}")

                # implicit viscosity                                                                                                                                                                                                
                u_star = self._helmholtz_face(rhs_u, nu)
                v_star = self._helmholtz_face(rhs_v, nu)
                assert torch.isfinite(u_star).all() and torch.isfinite(v_star).all(), f"NaN/Inf: Helmholtz (step {step})"

                # projection                                                                                                                                                                                                        
                div_star = self._div_f(u_star, v_star) / self.dt
                phi = self._poisson_center(div_star)
                gx, gy = self._grad_c2f(phi)
                u = u_star - self.dt * gx
                v = v_star - self.dt * gy
                assert torch.isfinite(u).all() and torch.isfinite(v).all(), f"NaN/Inf: after projection (step {step})"

                # filter after projection                                                                                                                                                                                           
                u = self._filter_faces(u); v = self._filter_faces(v)
                assert torch.isfinite(u).all() and torch.isfinite(v).all(), f"NaN/Inf: after post-proj filter (step {step})"

                if step == 0:
                    print(f"[dbg] step0 init max|u|={float(u.abs().max()):.3e} max|v|={float(v.abs().max()):.3e}")

            dv_dx = (torch.roll(v, -1, -2) - torch.roll(v, +1, -2)) / (2 * self.dx)
            du_dy = (torch.roll(u, -1, -1) - torch.roll(u, +1, -1)) / (2 * self.dx)
            w_next32 = dv_dx - du_dy

        out = w_next32.to(w.dtype)
        if added_batch:
            out = out.squeeze(0)
        return out

    # ---------------- spectral helpers (used by RK4 path) ----------------
    def _phys_params(self):
        # logRe is a learnable nn.Parameter in your code
        nu = torch.exp(-self.logRe)
        gamma = self.gamma
        F0 = self.F0
        return nu, gamma, F0
    
    def _bi_laplacian(self, f: Tensor) -> Tensor:
        fh = torch.fft.fft2(f, norm="forward")
        return torch.fft.ifft2(self.k4 * fh, norm="forward").real  # ∇⁴ f                                                                                                                                                           

    def _dealias_hat(self, f_hat: Tensor) -> Tensor:
        return f_hat * self.dealias_mask

    def _velocity_from_vorticity(self, w: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            w_hat = torch.fft.fft2(w.float(), norm="forward")
            k2 = (self.kx**2 + self.ky**2).float()
            k2 = k2.clone(); k2[0, 0] = 1.0                 # safe divide at DC                                                                                                                                                     
            psi_hat = -w_hat / k2
            psi_hat[..., 0, 0] = 0.0
            psi_hat = psi_hat * self.dealias_mask           # (optional but helps)                                                                                                                                                  
            u_hat = 1j * self.ky.float() * psi_hat
            v_hat = -1j * self.kx.float() * psi_hat
            u = torch.fft.ifft2(u_hat, norm="forward").real
            v = torch.fft.ifft2(v_hat, norm="forward").real
        return u, v

    def _laplacian(self, f: Tensor) -> Tensor:
        f_hat = torch.fft.fft2(f, norm="forward")
        return torch.fft.ifft2(self.laplacian * f_hat, norm="forward").real

    def _spectral_derivative(self, f: Tensor, axis: int) -> Tensor:
        f_hat = torch.fft.fft2(f, norm="forward")
        d_hat = 1j * (self.kx if axis == 0 else self.ky) * f_hat
        d_hat = self._dealias_hat(d_hat)          # dealias derivatives like JAX path                                                                                                                                               
        return torch.fft.ifft2(d_hat, norm="forward").real

    def _filter_faces(self, f):
        # 2/3 Orszag filter on face-defined field f, in fp32                                                                                                                                                                        
        with torch.cuda.amp.autocast(enabled=False):
            fh = torch.fft.fft2(f.float(), norm="forward")
            fh = fh * self.dealias_mask  # (N,N) -> broadcasts over batch                                                                                                                                                           
            out = torch.fft.ifft2(fh, norm="forward").real
        return out.to(f.dtype)

    # ---------------- PR-Smoother noise helpers ----------------                                                                                                                                                                   
    def sample(self, w: Tensor, *, n_steps: int = 1) -> Tensor:
        w_next = self.integrate(w, n_obs=n_steps)
        return w_next# + torch.randn_like(w_next) * self.sigma

    def log_prob(self, w_prev: Tensor, w_next: Tensor, *, n_obs: int = 1) -> Tensor:
        pred = self.integrate(w_prev, n_obs=n_obs)
        resid = (w_next - pred) / self.sigma
        logp = -0.5 * (resid.square() + _LOG2PI).flatten(1).sum(-1)
        logp -= resid.flatten(1).size(-1) * self.log_sigma
        return logp

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp().clamp(min=1e-100)
