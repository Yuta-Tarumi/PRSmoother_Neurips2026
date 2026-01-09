from __future__ import annotations
import argparse, math, pathlib, time
import numpy as np
from tqdm import tqdm

import jax, jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax.dlpack as jdl

import torch
from torch import nn
from torch.utils import dlpack as tdl

# --- your PyTorch dynamics (import the class you pasted) ---
from models.dynamics.Kolmogorov import KolmogorovDynamics  # adjust if needed

# ---------------- Constants (match your JAX driver) ----------------
N = 128
L = 2 * math.pi
RE = 1_000.0
NU = 1.0 / RE
GAMMA = 0.1
F0 = 1.0
K_FORCE = 4
DT = 5e-3
SPIN_UP = 4_000
KEEP_SNAPS = 10
BATCH = 96
ALPHA = 0.05

# ---------------- Helpers ----------------
def _to_jnp_array(x):
    """Extract raw jax array from GridVariable / GridArray / jnp array."""
    if hasattr(x, "data"):
        return x.data
    if hasattr(x, "array"):
        a = x.array
        return a.data if hasattr(a, "data") else a
    return x

def _jax_to_torch(a, device, dtype=torch.float32):
    """Zero-copy via DLPack, falls back to host copy if needed."""
    try:
        t = tdl.from_dlpack(jdl.to_dlpack(a))
    except Exception:
        t = torch.from_numpy(np.asarray(a))
    return t.to(device=device, dtype=dtype).contiguous()

def vorticity_from_uv_torch(u: torch.Tensor, v: torch.Tensor, L: float, N: int) -> torch.Tensor:
    dx = L / N
    dv_dx = (torch.roll(v, -1, 0) - torch.roll(v, 1, 0)) / (2 * dx)
    du_dy = (torch.roll(u, -1, 1) - torch.roll(u, 1, 1)) / (2 * dx)
    return dv_dx - du_dy

# put near the top (with other helpers)
def _as_2d(w: torch.Tensor) -> torch.Tensor:
    return w[0] if (w.ndim == 3 and w.size(0) == 1) else w

# ---------------- IC generator (JAX) ----------------
def jax_initial_w(seed: int, grid, device) -> torch.Tensor:
    key = jax.random.PRNGKey(seed)
    u_gv, v_gv = cfd.initial_conditions.filtered_velocity_field(
        key, grid=grid, maximum_velocity=3.0, peak_wavenumber=float(K_FORCE)
    )
    u_jnp = _to_jnp_array(u_gv)  # (N,N)
    v_jnp = _to_jnp_array(v_gv)  # (N,N)
    u_t = _jax_to_torch(u_jnp, device)
    v_t = _jax_to_torch(v_jnp, device)
    w0 = vorticity_from_uv_torch(u_t, v_t, L=L, N=N).to(torch.float32)
    return w0

# ---------------- One movie ----------------
@torch.no_grad()
def make_movie(
    base_seed: int,
    steps: int,
    save_every: int,
    device: torch.device,
    integrator: str,
    root: pathlib.Path,
) -> np.ndarray:
    # JAX grid for ICs
    grid = grids.Grid(shape=(N, N), domain=((0, L), (0, L)))

    # Dynamics in pure PyTorch
    dyn = KolmogorovDynamics(
        N=N, L=L, k_force=K_FORCE, internal_dt=DT, obs_dt=DT,  # obs_dt=DT → integrate(..., n_obs=1) = one dt
        gamma=GAMMA,
        device=device, dtype=torch.float32
    ).eval()

    # Ensure physical params match JAX driver
    dyn.F0 = nn.Parameter(torch.tensor(F0, device=device))
    dyn.logRe = nn.Parameter(torch.log(torch.tensor(RE, device=device, dtype=torch.float32)))

    # Initial vorticity from JAX
    w = jax_initial_w(base_seed, grid, device)
    w = _as_2d(w)  # ensure (N, N)

    # spin-up
    for _ in range(SPIN_UP):
        w = dyn.integrate(w, n_obs=1)
        w = _as_2d(w)

    # collect snapshots
    snaps = []
    for step in range(1, steps + 1):
        w = dyn.integrate(w, n_obs=1)
        w = _as_2d(w)
        if step % save_every == 0:
            snaps.append(w.detach().cpu().numpy().astype(np.float32))
            if len(snaps) == KEEP_SNAPS:
                break
    return np.stack(snaps, axis=0)

# ---------------- Batch writer ----------------
def save_batch(index: int, split: str, steps: int, save_every: int, device, integrator: str, root: pathlib.Path):
    outdir = root / split
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"seed{index:07d}.npz"
    if outfile.exists():
        print(f"[skip] {outfile}")
        return

    movies = np.empty((BATCH, KEEP_SNAPS, N, N), np.float32)
    base_seed = index * BATCH

    for i in tqdm(range(BATCH), desc=f"batch {index}", ncols=80):
        movies[i] = make_movie(
            base_seed=base_seed + i,
            steps=steps,
            save_every=save_every,
            device=device,
            integrator=integrator,
            root=root,
        )

    np.savez_compressed(outfile, w=movies)
    print(f"[done] {outfile}  -> {movies.nbytes/1e9:.2f} GB")

# ---------------- CLI ----------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate Kolmogorov-flow trajectories with JAX ICs and pure-PyTorch dynamics."
    )
    p.add_argument("--index", type=int, required=True, help="Batch index (global file id).")
    p.add_argument("--split", choices=["train", "test"],
                   help='Dataset split; defaults to "test" when --index==1_000_000, else "train".')
    p.add_argument("--steps", type=int, default=4400, help="Total dt-steps AFTER spin-up (default 4400).")
    p.add_argument("--save_every", type=int, default=10, help="Save every N dt-steps (default 10 → 0.1).")
    p.add_argument("--integrator", choices=["rk4"], default="rk4",
                   help="PyTorch dynamics integrator to use (default rk4).")
    p.add_argument("--root", type=pathlib.Path,
                   default=pathlib.Path("training_data/Kolmogorov"),
                   help="Root directory containing drag_all.txt and output folders.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help='torch device, e.g. "cuda", "cuda:0", or "cpu".')
    return p.parse_args()

def main():
    args = _parse_args()
    split = args.split or ("test" if args.index == 1_000_000 else "train")
    device = torch.device(args.device)

    tic = time.time()
    save_batch(args.index, split, args.steps, args.save_every, device, args.integrator, args.root)
    print(f"⏱ Finished in {(time.time() - tic)/60:.1f} min")

if __name__ == "__main__":
    main()
