import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Lorenz96Dataset(Dataset):
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self,
                 root: str,
                 seeds: range | list,
                 steps: int,
                 observation: str,
                 noise_std: float,
                 bias_factor: float,
                 block_size: int,
                 generate_4d_online: bool = False,
                 F: float = 8.0,
                 dt: float = 0.03,
                 internal_dt: float = 0.006
                 ):
        super().__init__()
        self.root = root
        self.steps = int(steps)
        self.seeds = list(seeds)
        self.observation = observation
        self.noise_std = float(noise_std)
        self.bias_factor = float(bias_factor)
        self.block_size = int(block_size)
        self.generate_4d_online = bool(generate_4d_online)
        self.F = float(F)
        self.dt = float(dt)
        self.internal_dt = float(internal_dt)
        self.num_each_step = int(self.dt//self.internal_dt)
        self.dtype = torch.float32

        # 40-D precomputed case still uses the bias file exactly as before.
        # In 4-D online mode we don't need it and we also avoid loading the file.
        if not self.generate_4d_online:
            self.bias = torch.zeros(40)
        else:
            self.bias = None

        # For 4-D online mode: figure out how many internal steps per outer dt
        if self.generate_4d_online:
            n_sub = self.dt / self.internal_dt
            if abs(round(n_sub) - n_sub) > 1e-6:
                raise ValueError(
                    f"dt={self.dt} must be an integer multiple of "
                    f"internal_dt={self.internal_dt}"
                )
            self.n_internal_steps = int(round(n_sub))
        else:
            self.n_internal_steps = None

    # ──────────────────────────────────────────────────────────────────────
    def _file_path(self, seed: int) -> str:
        return os.path.join(self.root, f"seed{seed:07d}.npz")

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        # one index == one file / one random seed
        return len(self.seeds)

    # ──────────────────────────────────────────────────────────────────────
    def _lorenz96_rhs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lorenz-96 RHS:
            dx_k/dt = (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F
        with cyclic boundary conditions on the last dimension.
        x: (B, D)
        """
        x_im2 = torch.roll(x, shifts=2, dims=-1)
        x_im1 = torch.roll(x, shifts=1, dims=-1)
        x_ip1 = torch.roll(x, shifts=-1, dims=-1)
        return (x_ip1 - x_im2) * x_im1 - x + self.F

    # ──────────────────────────────────────────────────────────────────────
    def _simulate_lorenz96(self, u0: torch.Tensor, T_total: int) -> torch.Tensor:
        """
        4th-order Runge–Kutta integrator for Lorenz-96.

        u0: (B, D)
        returns: (B, T_total, D)

        In 4-D online mode, each outer step of size `dt`
        is composed of `n_internal_steps` substeps of size `internal_dt`.
        """
        B, D = u0.shape
        device = u0.device
        x = u0
        traj = torch.empty((B, T_total, D), dtype=self.dtype, device=device)
        traj[:, 0, :] = x

        # In 4-D online mode this will be (5, 0.006) by default,
        # giving an effective outer step of 0.03.
        if self.n_internal_steps is not None:
            n_internal = self.n_internal_steps
            dt_internal = self.internal_dt
        else:
            # Fallback (not actually used in the 40-D case)
            n_internal = 1
            dt_internal = self.dt

        for t in range(1, T_total):
            for _ in range(n_internal):
                k1 = self._lorenz96_rhs(x)
                k2 = self._lorenz96_rhs(x + 0.5 * dt_internal * k1)
                k3 = self._lorenz96_rhs(x + 0.5 * dt_internal * k2)
                k4 = self._lorenz96_rhs(x + dt_internal * k3)
                x = x + (dt_internal / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            traj[:, t, :] = x

        return traj


    # ──────────────────────────────────────────────────────────────────────
    def _generate_4d_online(self, seed: int):
        """
        Generate a block of 4-D trajectories on-the-fly.

        Returns:
            truth : (block_size, steps, 4)
            obs   : (block_size, steps, 4)
        """
        B = self.block_size
        D = 4

        # match the same t_slice convention as the file-based code
        if self.steps == 1:
            T_total = 10   # need at least 10 steps to take t=9
            t_slice = slice(9, 10)
        else:
            T_total = self.steps
            t_slice = slice(0, self.steps)

        gen = torch.Generator(device="cpu").manual_seed(seed)

        # u0 ~ N(0, 2^2)
        u0 = torch.randn((B, D), dtype=self.dtype, device="cpu", generator=gen) * 2.0

        # deterministic dynamics
        truth_full = self._simulate_lorenz96(u0, T_total)  # (B, T_total, 4)
        truth = truth_full[:, t_slice, :]                   # (B, steps, 4)

        # Add Gaussian observation noise
        noise_full = torch.randn(
            (B, T_total, D),
            dtype=self.dtype,
            device="cpu",
            generator=gen,
        ) * self.noise_std
        noise = noise_full[:, t_slice, :]

        obs = truth * truth + noise

        return truth, obs, D

    # ──────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        seed = self.seeds[idx]

        # ---- 4-D on-the-fly mode ----------------------------------------
        if self.generate_4d_online:
            truth, obs, D = self._generate_4d_online(seed)

        # ---- Original file-based mode (40-D or 4-D precomputed) ---------
        else:
            path = self._file_path(seed)

            try:
                # (block_size, T, D) — cast once to float32 to avoid later surprises
                w = np.load(path)["x"].astype(np.float32, copy=False)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Missing file for seed {seed}: {path}") from e

            if w.shape[0] != self.block_size:
                raise ValueError(
                    f"{path} has shape {w.shape}, "
                    f"expected ({self.block_size}, 50, 40))."
                )

            B, T, D = w.shape

            # Select the time window (keep this logic exactly as you want it)
            if self.steps == 1:
                t_slice = slice(9, 10)   # use only t = 9
            else:
                t_slice = slice(0, self.steps)  # use t = 0..steps-1

            truth = torch.as_tensor(w[:, t_slice, :], dtype=self.dtype)

            # --- deterministic Gaussian noise tied to (seed, time) ---
            # Generate for ALL T, then slice, so overlapping times match across runs.
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise_full = torch.randn(
                (B, T, D),
                dtype=self.dtype,
                device="cpu",
                generator=gen,
            ) * self.noise_std
            noise = noise_full[:, t_slice, :].to(truth.device)

            if D == 40:
                if self.observation in ["full_linear", "sparse"]:
                    obs = truth + noise + self.bias * self.bias_factor
                if self.observation in ["full_nonlinear"]:
                    truth_to_2_clamped = torch.clamp(
                        (truth + self.bias * self.bias_factor) ** 2,
                        0,
                        5,
                    )
                    obs = truth_to_2_clamped + noise
                elif self.observation in ["square"]:
                    obs = truth * truth + noise + self.bias * self.bias_factor
                elif self.observation in ["full_pm3", "sparse_pm3"]:
                    obs = torch.clamp(
                        truth + self.bias * self.bias_factor,
                        min=-3.0,
                        max=3.0,
                    ) + noise
            elif D == 4:
                # old 4-D file-based behavior: x^2 + noise
                obs = truth * truth + noise
            else:
                raise ValueError(f"Unsupported state dimension D={D}")

        # ---- Common output formatting (same as original) -----------------
        if self.observation in ["full_linear", "full_nonlinear", "full_pm3", "square", "biased_abs"]:
            return {"truth": truth, "obs": obs}
        elif self.observation in ["sparse", "sparse_pm3"]:
            return {"truth": truth, "obs": obs[..., ::4]}
        else:
            raise ValueError(f"Unknown observation type: {self.observation!r}")


# ──────────────────────────────────────────────────────────────────────
# Optional helper to flatten (N, block_size, T, D) → (N*block_size, T, D)
def collate_flatten(batch):
    """Custom collate_fn for DataLoader if you want a fully-flat batch."""
    truth = torch.cat([item["truth"] for item in batch], dim=0)
    obs = torch.cat([item["obs"] for item in batch], dim=0)
    return {"truth": truth, "obs": obs}
