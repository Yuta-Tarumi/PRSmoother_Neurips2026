import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def observe_coarse(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, kernel_size=8, stride=8)

class KolmogorovDataset(Dataset):
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self,
                 root: str = "/work/go84/o84000/training_data/Kolmogorov/train",
                 seeds: range | list | None = None,
                 bias_seed: int = 0,
                 observation: str = "full",
                 noise_std: float = 1.0,
                 bias_factor: float = 1.0,
                 block_size: int = 96,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.root = root
        self.seeds = list(seeds) if seeds is not None else list(range(16_384))
        self.bias = torch.tensor(np.load(f"/work/go84/o84000/training_data/Kolmogorov/offset_seed{bias_seed}.npy"), dtype=torch.float32)
        self.observation = observation
        self.noise_std = float(noise_std)
        self.bias_factor = float(bias_factor)
        self.block_size = int(block_size)
        self.dtype = dtype

    # ──────────────────────────────────────────────────────────────────────
    def _file_path(self, seed: int) -> str:
        return os.path.join(self.root, f"seed{seed:07d}.npz")

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        # one index == one file
        return len(self.seeds)

    # ──────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        seed = self.seeds[idx]
        path = self._file_path(seed)
        #print(f"{path=}")
        try:
            # (block_size, T, D) — cast once to float32 to avoid later surprises
            w = np.load(path)["w"].astype(np.float32, copy=False)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing file for seed {seed}: {path}") from e

        if w.shape[0] != self.block_size:
            raise ValueError(f"{path} has shape {w.shape}, "
                             f"expected ({self.block_size}, 10, 128, 128)).")
        
        truth = torch.as_tensor(w, dtype=self.dtype)           # (B, T, D, D)
        # --- deterministic Gaussian noise -------------------------------
        gen = torch.Generator().manual_seed(seed)
        if self.observation in ["full", "half"]:
            noise = torch.randn(truth.shape,
                                dtype=self.dtype,
                                device=truth.device,
                                generator=gen) * self.noise_std
            obs = truth + self.bias*self.bias_factor + noise
        elif self.observation in ["coarse"]:
            noise = torch.randn(observe_coarse(truth).shape,
                                dtype=self.dtype,
                                device=truth.device,
                                generator=gen) * self.noise_std
            obs = observe_coarse(truth + self.bias*self.bias_factor) + noise
        elif self.observation in ["sparse"]:
            base = truth + self.bias * self.bias_factor
            obs = torch.full_like(truth, float("nan"))
            lr = base[..., 3::8, 3::8]  # (B, T, 16, 16)
            noise_lr = torch.randn(lr.shape, dtype=self.dtype, device=truth.device, generator=gen) * self.noise_std
            obs[..., 3::8, 3::8] = lr + noise_lr
        elif self.observation in ["mixed"]:
            base = truth + self.bias * self.bias_factor

            # Start with all-NaN canvas to mark unobserved locations
            obs = torch.full_like(truth, float("nan"))

            # Low-res sparse samples at positions 3 mod 8
            lr = base[..., 3::8, 3::8]  # (B, T, 16, 16)
            noise_lr = torch.randn(lr.shape, dtype=self.dtype, device=truth.device, generator=gen) * self.noise_std
            obs[..., 3::8, 3::8] = lr + noise_lr

            # High-res full block in top-left 64x64
            hr = base[..., :64, :64]  # (B, T, 64, 64)
            noise_hr = torch.randn(hr.shape, dtype=self.dtype, device=truth.device, generator=gen) * self.noise_std
            obs[..., :64, :64] = hr + noise_hr  # overwrite LR overlaps with HR        
        else:
            print(f"unknown {self.observation=}")
            return 1

        if self.observation in ["full"]:
            return {"truth": truth[:, :10], "obs": obs[:, :10]}
        elif self.observation in ["half"]:
            return {"truth": truth[:, :10, :, :], "obs": obs[:, :10, :64, :]}
        elif self.observation in ["coarse"]:
            return {"truth": truth, "obs": obs}
        elif self.observation in ["sparse"]:
            return {"truth": truth, "obs": obs}
        elif self.observation in ["mixed"]:
            return {"truth": truth[:, :10], "obs": obs[:, :10]}


        
