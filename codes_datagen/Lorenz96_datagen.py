
from __future__ import annotations
import argparse, os, pathlib, numpy as np
import torch
from typing import Tuple
from tqdm import tqdm
from models.dynamics.Lorenz96 import L96Dynamics, integrate_l96

# ---------- Lorenz-96 dynamics ----------
F = 8.0
DT_INT = 0.003
DT_SNAP = 0.03
STEPS_PER_SNAP = int(round(DT_SNAP / DT_INT))  # 10
N_VARS = 40
TOTAL_SNAPS = 100
SPINUP_SNAPS = 50
TRAJ_LEN = TOTAL_SNAPS - SPINUP_SNAPS            # 50
BATCH_SIZE = 1024

def advance_one_snapshot(x: torch.tensor) -> torch.tensor:
    return integrate_l96(x, F=torch.tensor(F), obs_dt=DT_SNAP, internal_dt=DT_INT)
"""
    for _ in range(STEPS_PER_SNAP):
        x = integrate_l96(x, F=torch.tensor(F), obs_dt=DT_SNAP, internal_dt=DT_INT)
    return x
"""
def generate_trajectory(rng: np.random.Generator) -> np.ndarray:
    x = torch.tensor(rng.random(N_VARS) * 20.0 - 10.0)          # U[-10,10]
    traj = np.empty((TOTAL_SNAPS, N_VARS))
    for s in range(TOTAL_SNAPS):
        x = advance_one_snapshot(x)
        traj[s] = x
    return traj[SPINUP_SNAPS:]                    # (50,40)

# ---------- I/O helpers ----------
def _ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def save_batch(split: str, index: int, root: pathlib.Path):
    """Generate and save one NPZ batch (1024 trajectories)."""
    out_dir = root / "Lorenz96" / split
    _ensure_dir(out_dir)

    out_file = out_dir / f"seed{index:07d}.npz"
    if out_file.exists():
        print(f"[skip] {out_file} already exists")
        return

    base_seed = index * BATCH_SIZE
    batch = np.empty((BATCH_SIZE, TRAJ_LEN, N_VARS), dtype=np.float32)

    for i in tqdm(range(BATCH_SIZE), desc=f"batch {index}", ncols=80):
        seed_id = base_seed + i
        traj = generate_trajectory(np.random.default_rng(seed_id))
        batch[i] = traj

    np.savez_compressed(out_file, x=batch.astype(np.float32))
    print(f"[done] wrote {out_file}")

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Generate Lorenz-96 batches")
    p.add_argument("--index", type=int, required=True,
                   help="Batch index (one per Slurm job)")
    p.add_argument("--split", type=str, choices=["train", "test"],
                   help="Dataset split.  If omitted, index 1e6 ⇒ test else train")
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("training_data"),
                   help="Project data root (default: training_data)")
    args = p.parse_args()

    split = args.split or ("test" if args.index == 1_000_000 else "train")
    save_batch(split, args.index, args.root)

if __name__ == "__main__":
    main()
