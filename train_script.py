
from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import random
import os

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
torch.backends.cuda.matmul.allow_tf32 = True

from models.PRSmoother import PRSmoother
from models.mean_field import MFSmoother
from datasets.Lorenz96 import Lorenz96Dataset, collate_flatten
from datasets.Kolmogorov import KolmogorovDataset
# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_ini(path: str | Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f"Could not load config at: {path}")
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Train Physics‑Rollout Smoother on Lorenz‑96")
    parser.add_argument("--config", required=True, help="Path to INI/TOML config file")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # 1) config -------------------------------------------------------------
    cfg_path = Path(args.config).resolve()
    dataset_name = cfg_path.parent.stem       # e.g. "Lorenz96"
    cfg_name     = cfg_path.stem              # e.g. "baseline"

    cfg = load_ini(cfg_path)
    print(f"{cfg=}")
    
    observation   = cfg.get("data", "observation", fallback="full")
    noise_std     = cfg.getfloat("data", "noise", fallback=1.0)
    bias_factor   = cfg.getfloat("data", "bias_factor", fallback=1.0)
    bias_seed     = cfg.getint("data", "bias_seed", fallback=42)

    steps         = cfg.getint("data", "steps", fallback=50)
    print(f"{bias_factor=}")
    root          = cfg.get("data", "root", fallback=f"training_data/{dataset_name}/train")
    root_test     = cfg.get("data", "root_test", fallback=f"training_data/{dataset_name}/test")
    method        = cfg.get("model", "method", fallback="PR-Smoother")
    noisy_dyn     = cfg.get("model", "noisy_dynamics", fallback="False")
    latent_dim    = cfg.getint("model", "latent_dim", fallback=40)
    architecture  = cfg.get("model", "architecture", fallback="Conv1D")
    posterior     = cfg.get("model", "posterior", fallback="flow")
    lr            = cfg.getfloat("model", "lr", fallback=1e-3)
    boost_factor  = cfg.getfloat("model", "boost_factor", fallback=1.0)
    logstd_factor = cfg.getfloat("model", "logstd_factor", fallback=1.0)
    init_F        = cfg.getfloat("model", "init_F", fallback=0.0)
    train_F       = cfg.get("model", "train_F", fallback="T")
    train_Q       = cfg.get("model", "train_Q", fallback="T")
    train_R       = cfg.get("model", "train_R", fallback="T")
    init_obs_var  = cfg.getfloat("model", "init_obs_var")
    norm_factor   = cfg.getfloat("model", "norm_factor", fallback=10.0)
    sigma_0       = cfg.getfloat("model", "sigma_0", fallback=10.0)
    batch         = cfg.getint("model", "batch", fallback=64)
    epochs        = cfg.getint("model", "epochs", fallback=20)
    seed          = cfg.getint("model", "seed", fallback=42)
    train_St      = cfg.get("model", "train_St", fallback="False")
    
    set_seed(seed)

    # 2) dataset ------------------------------------------------------------
    if dataset_name in ["Lorenz96"]:
        steps = 50
        print(f"{dataset_name=}")
        dataset = "Lorenz96"
        warmup_epochs = 0.05
        train_ds = Lorenz96Dataset(root=root, seeds=range(0, 1), steps=50, observation=observation, block_size=1024, noise_std=noise_std, bias_factor=bias_factor)
        loader = DataLoader(train_ds, batch_size=batch, shuffle=True, persistent_workers=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
        test_ds     = Lorenz96Dataset(root=root_test, seeds=range(1_000_000, 1_000_001), steps=50, observation=observation, block_size=1024, noise_std=noise_std, bias_factor=bias_factor)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, persistent_workers=True,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
    elif dataset_name in ["Lorenz96_noisy"]:
        steps = 50
        print(f"{dataset_name=}")
        dataset = "Lorenz96"
        warmup_epochs = 0.05
        train_ds = Lorenz96Dataset(root=root, seeds=range(0, 1), steps=50, observation=observation, block_size=1024, noise_std=noise_std, bias_factor=bias_factor)
        loader = DataLoader(train_ds, batch_size=batch, shuffle=True, persistent_workers=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
        test_ds     = Lorenz96Dataset(root=root_test, seeds=range(1_000_000, 1_000_001), steps=50, observation=observation, block_size=1024, noise_std=noise_std, bias_factor=bias_factor)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, persistent_workers=True,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
    elif dataset_name in ["Lorenz96_multimodal", "Lorenz96_multimodal_4dim"]:
        print(f"{dataset_name=}")
        if dataset_name in ["Lorenz96_multimodal"]:
            dataset = "Lorenz96"
        else:
            dataset = "Lorenz96_4dim"
        warmup_epochs = 0.05
        train_ds = Lorenz96Dataset(root=root, seeds=range(0, 10000), steps=steps, observation="square", block_size=1024, noise_std=noise_std, bias_factor=bias_factor, generate_4d_online=True)
        loader = DataLoader(train_ds, batch_size=batch, shuffle=True, persistent_workers=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
        test_ds     = Lorenz96Dataset(root=root_test, seeds=range(1_000_000, 1_000_001), steps=steps, observation="square", block_size=1024, noise_std=noise_std, bias_factor=bias_factor, generate_4d_online=True)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, persistent_workers=True,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
    elif dataset_name in ["Kolmogorov"]:
        steps = 10
        print(f"{dataset_name=}")
        dataset = "Kolmogorov"
        warmup_epochs = 0.05
        train_ds = KolmogorovDataset(root=root, seeds=range(0, 1), observation=observation, noise_std=noise_std, bias_factor=bias_factor, bias_seed=bias_seed)
        print(f"{train_ds=}")
        loader = DataLoader(train_ds, batch_size=batch, shuffle=True, persistent_workers=True,
                                      num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
        test_ds     = KolmogorovDataset(root=root_test, seeds=range(1_000_000, 1_000_001), observation=observation, noise_std=noise_std, bias_factor=bias_factor, bias_seed=bias_seed)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, persistent_workers=True,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=collate_flatten)
    else:
        print(f"Unknown dataset: {dataset_name}")
        return 0

    print(f"{dataset_name=}")
    # 3) model --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_F in ["T"]:
        train_F = True
    elif train_F in ["F"]:
        train_F = False
    else:
        print(f"train_F must be T or F, {train_F=}")
        return 1
    if train_Q in ["T"]:
        train_Q = True
    elif train_Q in ["F"]:
        train_Q = False
    else:
        print(f"train_Q must be T or F, {train_Q=}")
        return 1
    if train_R in ["T"]:
        learn_obs_var = True
    elif train_R in ["F"]:
        learn_obs_var = False
    else:
        print(f"train_R must be T or F, {train_R=}")
        return 1

    if method in ["PR-Smoother"]:
        if noisy_dyn == "True":
            noisy = True
        elif noisy_dyn == "False":
            noisy = False
        else:
            print(f"undefined {noisy_dyn=}")
            return 1 
        model  = PRSmoother(
            dataset=dataset,
            latent_dim=latent_dim,
            init_F=init_F,
            train_F=train_F,
            train_Q=train_Q,
            observation=observation,
            norm_factor=norm_factor,
            sigma0_sq=sigma_0**2,
            encoder=architecture,
            steps=steps,
            posterior=posterior,
            learn_obs_var=learn_obs_var,
            obs_var=init_obs_var,
            noisy=noisy,
        ).to(device)
    elif method in ["mean-field"]:
        # placeholder
        model = MFSmoother(
            dataset=dataset,
            latent_dim=latent_dim,
            init_F=init_F,
            train_F=train_F,
            train_Q=train_Q,
            observation=observation,
            norm_factor=norm_factor,
            sigma0_sq=sigma_0**2,
            encoder=architecture,
            steps=steps,
            learn_obs_var=learn_obs_var,
            obs_var=init_obs_var,
        ).to(device)
    else:
        print(f"unknown {method=}")
        return 1

    boosted_params = []   # dyn.* but NOT dyn.log_sigma
    log_sigma_param = []
    base_params    = []   # everything else
    
    for name, param in model.named_parameters():
        if name.startswith("dyn."):
            if name.endswith(""):
                log_sigma_param.append(param)
            else:
                boosted_params.append(param)
        elif name.endswith("log_r"):
            print(f"{name=}")
            boosted_params.append(param)
        elif name.endswith("_log_dyn_noise_raw"):
            print(f"{name=}")
            log_sigma_param.append(param)
        elif name.endswith("_log_gen_dyn_noise_raw"):
            print(f"{name=}")
            log_sigma_param.append(param)
        else:
            base_params.append(param)
    print(f"{log_sigma_param=}")
    optim = torch.optim.Adam(
        [
            {"params": boosted_params, "lr": boost_factor * lr},
            {"params": log_sigma_param, "lr": logstd_factor * lr},
            {"params": base_params}                     # inherits default lr
        ],
        lr=lr
    )
    print(f"{boosted_params=}")

    if dataset_name in ["Lorenz96", "Lorenz96_noisy", "Lorenz96_multimodal", "Lorenz96_multimodal_4dim"]:
        warmup_iters = int(10000*0.01*epochs//batch)
        decay_iters  = int(10000*0.99*epochs//batch)       # epochs left after warm-up
        final_factor = 0.1
    elif dataset_name in ["Kolmogorov"]:
        warmup_iters = int(30000*0.05*epochs//batch)
        decay_iters  = int(30000*0.95*epochs//batch)       # epochs left after warm-up
        final_factor = 0.1

    warmup_sched = LinearLR(
        optim,
        start_factor=0.001,        # 0 means lr == 0 on the very first step
        end_factor=1.0,
        total_iters=warmup_iters,
    )

    gamma = final_factor ** (1.0 / decay_iters)
    decay_sched = ExponentialLR(optim, gamma=gamma)

    print(f"{warmup_epochs=}, {warmup_iters=}, {decay_iters=}")

    scheduler = SequentialLR(
        optim,
        schedulers=[warmup_sched, decay_sched],
        milestones=[warmup_iters]   # switch after the warm-up epochs are done
    )
    
    #scheduler = ExponentialLR(optim, gamma=gamma)
    
    ckpt_dir = Path("output") / dataset_name / cfg_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path("output") / dataset_name / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(f"{out_dir}/outputs")
    os.makedirs(f"{output_dir}", exist_ok=True)
    elbo_file    = (out_dir / "ELBO.txt").open("w")
    elbo1_file   = (out_dir / "ELBO1.txt").open("w")
    elbo2_file   = (out_dir / "ELBO2.txt").open("w")
    elbo3_file   = (out_dir / "ELBO3.txt").open("w")
    if dataset_name in ["Lorenz96", "Lorenz96_noisy", "Lorenz96_multimodal", "Lorenz96_multimodal_4dim"]:
        F0_file      = (out_dir / "F.txt").open("w")
        logR_file    = (out_dir / "logR.txt").open("w")
    elif dataset_name in ["Kolmogorov"]:
        F0_file      = (out_dir / "F0.txt").open("w")
        logRe_file   = (out_dir / "logRe.txt").open("w")
        logR_file    = (out_dir / "logR.txt").open("w")
        log_alpha_file    = (out_dir / "log_alpha.txt").open("w")
        log_beta_file    = (out_dir / "log_beta.txt").open("w")
        log_gamma_file    = (out_dir / "log_gamma.txt").open("w")
        boost_factor_k_wall_file   = (out_dir / "boost_factor_k_wall.txt").open("w")
        
    sigma_file  = (out_dir / "sigma_dyn.txt").open("w")
    test_elbo_file   = (out_dir / "TEST_ELBO.txt").open("w")
    test_rmse_file   = (out_dir / "TEST_RMSE.txt").open("w")

    @torch.no_grad()
    def evaluate(model, loader, suffix=""):
        if suffix != "":
            print(f"evaluating {suffix=}")
        model.eval()
        rmse_sum_t = None        # accumulate Σ_RMSE over B for each t
        n_samples  = 0           # total B across all batches
        loss_sum   = 0.0         # accumulate −ELBO × B

        for batch_data in loader:
            obs   = batch_data["obs"].to(device)      # (B, T, D)
            truth = batch_data["truth"].to(device)    # (B, T, D)
            
            # model returns (−ELBO, xs)  →  first is loss, second is prediction
            #with autocast(dtype=torch.bfloat16):
            #with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            neg_elbo, xs, elbo1, elbo2, elbo3 = model(obs)

            # 1. mean over D, 2. sqrt  → RMSE per sample & time (B, T)
            rmse_bt = torch.sqrt(torch.mean((xs - truth) ** 2, dim=-1))
            
            # 3. accumulate for mean over B later
            if rmse_sum_t is None:
                rmse_sum_t = rmse_bt.sum(dim=0)      # (T,)
            else:
                rmse_sum_t += rmse_bt.sum(dim=0)

            n_samples += rmse_bt.size(0)
            loss_sum  += neg_elbo.item() * rmse_bt.size(0)

            torch.save(obs.detach(), f"{output_dir}/obs{suffix}")
            torch.save(truth.detach(), f"{output_dir}/truth{suffix}")
            torch.save(xs.detach(), f"{output_dir}/xs{suffix}")
            
        model.train()

        rmse_t   = rmse_sum_t / n_samples if n_samples else torch.full((0,), float("nan"))
        avg_loss = loss_sum  / n_samples if n_samples else float("nan")
        return avg_loss, rmse_t

   # 5) train loop ---------------------------------------------------------
    write_every = 20  # batches
    for epoch in range(1, epochs + 1):
        step = 0
        if dataset_name in ["Kolmogorov"]:
            if observation in ["full", "mixed", "sparse"]:
                bias_arr  = np.empty((int(30000//(write_every*batch)), 128, 128), dtype=np.float32)
            elif observation in ["half"]:
                bias_arr  = np.empty((int(30000//(write_every*batch)), 64, 128), dtype=np.float32)
            elif observation in ["coarse"]:
                bias_arr  = np.empty((int(30000//(write_every*batch)), 16, 16), dtype=np.float32)
        elif dataset_name in ["Lorenz96", "Lorenz96_multimodal", "Lorenz96_noisy"]:
            bias_arr  = np.empty((int(2000//write_every), 40), dtype=np.float32)
        elif dataset_name in ["Lorenz96_multimodal_4dim"]:
            bias_arr  = np.empty((int(1000//write_every), 4), dtype=np.float32)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_data in pbar:
            y = batch_data["obs"].to(device)
            #print(f"{y=}")
            with autocast(dtype=torch.bfloat16):
                #with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                loss, _, elbo1, elbo2, elbo3 = model(y)

            if (loss == 0) and (elbo1 == 0):
                optim.zero_grad(set_to_none=True)
                del _
                print(f"sample skipped due to divergence in dynamics")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); optim.zero_grad()

            # metrics
            if dataset_name in ["Lorenz96", "Lorenz96_noisy", "Lorenz96_multimodal", "Lorenz96_multimodal_4dim"]:
                elbo_val  = loss.item()
                elbo1_val     = elbo1.item()
                elbo2_val     = elbo2.item()
                elbo3_val     = elbo3.item()
                F_val     = model.dyn.F.item()
                logR_val      = model.log_r.item()
                log_sigma_val = model.dyn.log_sigma.item()
                bias_val      = model.bias.detach().cpu().numpy()
                log_dyn_noise = model.log_dyn_noise.item()
                log_prc_noise = model._log_process_noise_std.item()

                # progress bar update
                if noisy:
                    pbar.set_postfix({"−ELBO": f"{elbo_val:.4f}",
                                      "F": f"{F_val:.3f}",
                                      "log_dyn_noise": f"{log_dyn_noise:.3f}",
                                      "log_prc_noise": f"{log_prc_noise:.3f}"})
                else:
                    pbar.set_postfix({"−ELBO": f"{elbo_val:.4f}",
                                      "F": f"{F_val:.3f}"})
            elif dataset_name in ["Kolmogorov"]:
                elbo_val      = loss.item()
                elbo1_val     = elbo1.item()
                elbo2_val     = elbo2.item()
                elbo3_val     = elbo3.item()
                F0_val        = model.dyn.F0.item()
                logRe_val     = model.dyn.logRe.item()
                logR_val      = model.log_r.item()
                log_sigma_val = model.dyn.log_sigma.item()
                bias_val      = model.bias.detach().cpu().numpy()
                log_alpha     = model.log_alpha.detach().cpu().numpy()
                log_beta      = model.log_beta.detach().cpu().numpy()
                log_gamma     = model.log_gamma.detach().cpu().numpy()
                boost_kwall   = model.log_raw_boost.detach().cpu().numpy()

                # progress bar update
                pbar.set_postfix({"−ELBO": f"{elbo_val:.4f}",
                                  "logR": f"{logR_val:.3f}",
                                  "logRe": f"{logRe_val:.3f}",
                                  "F0": f"{F0_val:.3f}"})
                
                
            # write every N batches
            if step % write_every == 0:
                if dataset_name in ["Lorenz96", "Lorenz96_noisy", "Lorenz96_multimodal", "Lorenz96_multimodal_4dim"]:
                    elbo_file.write(f"{elbo_val}\n"); elbo_file.flush()
                    elbo1_file.write(f"{elbo1_val}\n"); elbo1_file.flush()
                    elbo2_file.write(f"{elbo2_val}\n"); elbo2_file.flush()
                    elbo3_file.write(f"{elbo3_val}\n"); elbo3_file.flush()
                    logR_file.write(f"{logR_val}\n"); logR_file.flush()
                    F0_file.write(f"{F_val}\n");     F0_file.flush()
                    sigma_file.write(f"{log_sigma_val}\n"); sigma_file.flush()
                    bias_arr[int(step//write_every)] = bias_val
                elif dataset_name in ["Kolmogorov"]:
                    elbo_file.write(f"{elbo_val}\n"); elbo_file.flush()
                    elbo1_file.write(f"{elbo1_val}\n"); elbo1_file.flush()
                    elbo2_file.write(f"{elbo2_val}\n"); elbo2_file.flush()
                    elbo3_file.write(f"{elbo3_val}\n"); elbo3_file.flush()
                    F0_file.write(f"{F0_val}\n");     F0_file.flush()
                    logRe_file.write(f"{logRe_val}\n");     logRe_file.flush()
                    sigma_file.write(f"{log_sigma_val}\n"); sigma_file.flush()
                    logR_file.write(f"{logR_val}\n"); logR_file.flush()
                    log_alpha_file.write(f"{log_alpha}\n"); log_alpha_file.flush()
                    log_beta_file.write(f"{log_beta}\n"); log_beta_file.flush()
                    log_gamma_file.write(f"{log_gamma}\n"); log_gamma_file.flush()
                    boost_factor_k_wall_file.write(f"{boost_kwall}\n"); boost_factor_k_wall_file.flush()
                    bias_arr[int(step//write_every)] = bias_val
                
            step += 1
            scheduler.step()
            #lr_now = optim.param_groups[0]['lr']
            #print(f"{optim.param_groups[0]['lr']=}, {optim.param_groups[1]['lr']=}, {optim.param_groups[2]['lr']=}")

        # ───── AFTER each epoch ────────────────────────────────────────────
        np.save((out_dir / f"bias_epoch{epoch}"), bias_arr)
        # Checkpoint & periodic test  # NEW ▼
        if epoch % 1 == 0 or epoch == epochs:
            
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "cfg": cfg_path.as_posix(),
                    "seed": seed,
                },
                ckpt_dir / f"epoch{epoch:07d}.pt",
            )
            
            if ((dataset_name in ["Lorenz96_multimodal", "Lorenz96_multimodal_4dim"]) and epoch == epochs):
                for i in range(1000):
                    test_elbo, test_rmse = evaluate(model, test_loader, suffix=f"{i:05d}")
            else:
                test_elbo, test_rmse = evaluate(model, test_loader)
            test_elbo_file.write(f"{epoch}\t{test_elbo}\n"); test_elbo_file.flush()
            rmse_cpu = test_rmse.cpu()

    # close files -----------------------------------------------------------
    elbo_file.close(); F0_file.close(); sigma_file.close(); test_elbo_file.close(), test_rmse_file.close()

if __name__ == "__main__":
    main()
