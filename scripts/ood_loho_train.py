"""Leave-One-Host-Out (LOHO) and Block-Out OOD experiments.

Implements:
  P0: Leave-one-G6-host-out (7-fold)
  P1: G6×3d compositional block-out

Uses v4 recipe: MAE loss, warmup, cosine, ct-UAE, 150ep, SWA.

Usage:
  # P0: leave one host out
  python scripts/ood_loho_train.py --exp p0 --fold MoS2 --gpu 0 --seed 42

  # P1: block-out G6 × 3d-TM
  python scripts/ood_loho_train.py --exp p1 --gpu 0 --seed 42

  # Run all P0 folds (launcher)
  python scripts/ood_loho_train.py --exp p0 --all --gpus 0,1,6
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import collate_fn  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

# ── Config ───────────────────────────────────────────────────────────────
EPOCHS = 150
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
SWA_START = 120
SWA_LR = 1e-4
LABEL_NOISE_STD = 0.03
GRAD_CLIP = 5.0
PATIENCE = 30

MODEL_KWARGS = dict(
    atom_fea_len=9, hidden_dim=128, n_local_layers=3, n_global_layers=2,
    num_heads=4, rcut_local=5.0, dmax_global=12.0, defect_embedding=True,
    dropout=0.1, ct_uae_path=str(ROOT / "data" / "ct_uae_mt3_embeddings.pt"),
)

# Flag to disable ct-UAE (set via --no-uae argument)
USE_UAE = True

# ── Experiment definitions ───────────────────────────────────────────────
G6_HOSTS = ['MoS2', 'MoSe2', 'MoTe2', 'WS2', 'WSe2', 'WTe2', 'MoSSe']

TM_3D = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

RESULTS = ROOT / "results" / "ood"


# ── Dataset with leave-out support ───────────────────────────────────────
class LOHODataset(Dataset):
    """Wraps a list of pre-processed samples (from cleaned_dataset.pkl)."""

    def __init__(self, samples: list):
        self.data = samples
        # Load atom features
        ref_path = ROOT / "data" / "atom_features_ref.pth"
        self.atom_features = torch.load(ref_path, map_location="cpu", weights_only=True)
        # Compute defect masks
        from ase.data import atomic_numbers as AZ
        for s in self.data:
            if "defect_mask" not in s:
                mask = np.zeros(len(s["numbers"]), dtype=np.int64)
                dopant = s["metadata"].get("dopant", "")
                z = AZ.get(dopant, None)
                if z is not None:
                    cands = np.flatnonzero(s["numbers"] == z)
                    if cands.size > 0:
                        mask[cands[-1]] = 1
                s["defect_mask"] = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        numbers = torch.from_numpy(s["numbers"])
        x = self.atom_features[numbers]
        return {
            "x": x,
            "atomic_numbers": numbers,
            "defect_mask": torch.from_numpy(s["defect_mask"]).long(),
            "edge_index": torch.from_numpy(s["edge_index"]),
            "edge_dist": torch.from_numpy(s["edge_dist"]),
            "edge_offset": torch.from_numpy(s["edge_offset"]).float(),
            "triplet_index": torch.from_numpy(s["triplet_index"]),
            "angles": torch.from_numpy(s["angles"]),
            "dist_matrix": torch.from_numpy(s["dist_matrix"]),
            "positions": torch.from_numpy(s["positions"]),
            "cell": torch.from_numpy(s["cell"]),
            "target": torch.tensor(s["target"], dtype=torch.float32),
            "num_atoms": numbers.numel(),
        }


def load_data():
    """Load IMP2D cleaned dataset."""
    path = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} samples from {path.name}")
    return data


def make_loho_split(data, holdout_hosts=None, holdout_dopants=None,
                    block_hosts=None, block_dopants=None, seed=42):
    """Create leave-out splits.

    Args:
        holdout_hosts: list of hosts to hold out entirely
        holdout_dopants: list of dopants to hold out entirely
        block_hosts + block_dopants: hold out the cross product

    Returns:
        train_samples, val_samples, test_samples
    """
    test_samples = []
    rest_samples = []

    for s in data:
        host = s["metadata"].get("host", "")
        dopant = s["metadata"].get("dopant", "")

        is_test = False
        if holdout_hosts and host in holdout_hosts:
            is_test = True
        if holdout_dopants and dopant in holdout_dopants:
            is_test = True
        if block_hosts and block_dopants:
            if host in block_hosts and dopant in block_dopants:
                is_test = True

        if is_test:
            test_samples.append(s)
        else:
            rest_samples.append(s)

    # Split rest into train/val (90/10)
    rng = random.Random(seed)
    rng.shuffle(rest_samples)
    n_val = max(1, int(0.1 * len(rest_samples)))
    val_samples = rest_samples[:n_val]
    train_samples = rest_samples[n_val:]

    return train_samples, val_samples, test_samples


# ── Training ─────────────────────────────────────────────────────────────
def train_model(train_loader, val_loader, device, seed, tag, model_kwargs=None):
    """Train CrystalTransformer with v4 recipe."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = model_kwargs if model_kwargs is not None else MODEL_KWARGS
    model = CrystalTransformer(**kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params/1e6:.4f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Warmup + cosine scheduler
    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        progress = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=SWA_LR)

    best_val, best_state, bad = float("inf"), None, 0
    history = []
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        losses = []
        use_swa = ep >= SWA_START

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            target = batch["target"]
            # Label noise
            if LABEL_NOISE_STD > 0:
                target = target + torch.randn_like(target) * LABEL_NOISE_STD

            pred = model(batch)
            loss = (pred - target).abs().mean()  # MAE loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(loss.item())

        # SWA update
        if use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                pred = model(batch)
                val_preds.append(pred.cpu())
                val_targets.append(batch["target"].cpu())
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_mae = (val_preds - val_targets).abs().mean().item()

        improved = ""
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            improved = "  *"
        else:
            bad += 1

        lr_now = optimizer.param_groups[0]["lr"]
        if ep % 10 == 0 or improved or ep == EPOCHS - 1:
            print(f"  ep {ep:>3}  loss={np.mean(losses):.4f}  val_mae={val_mae:.4f}  lr={lr_now:.2e}{improved}")

        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                       "val_mae": float(val_mae), "lr": lr_now})

        if bad >= PATIENCE and ep >= SWA_START:
            print(f"  Early stop at ep {ep} (patience={PATIENCE})")
            break

    # Final: use SWA model
    if SWA_START < EPOCHS:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_state = {k.replace("module.", ""): v for k, v in swa_model.state_dict().items()
                     if not k.startswith("n_averaged")}
        model.load_state_dict(swa_state, strict=False)
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                val_preds.append(model(batch).cpu())
                val_targets.append(batch["target"].cpu())
        swa_val = (torch.cat(val_preds) - torch.cat(val_targets)).abs().mean().item()
        print(f"  SWA val_mae={swa_val:.4f} vs best={best_val:.4f}")
        if swa_val < best_val:
            best_val = swa_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    wall = (time.time() - t0) / 60
    print(f"  Training done in {wall:.1f} min")

    # Load best state for evaluation
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_val, wall, history


def evaluate(model, test_loader, device):
    """Evaluate model on test set, return per-sample predictions."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model(batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    return preds, targets, mae, rmse


# ── Main ─────────────────────────────────────────────────────────────────
def run_single_fold(exp, fold, seed, gpu, no_uae=False):
    """Run a single fold of the OOD experiment."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    uae_label = " [NO-UAE ablation]" if no_uae else ""
    print(f"\n{'='*60}")
    print(f"OOD Experiment: {exp}, Fold: {fold}, Seed: {seed}, Device: {device}{uae_label}")
    print(f"{'='*60}")

    # Apply UAE ablation
    model_kwargs = dict(MODEL_KWARGS)
    if no_uae:
        model_kwargs["ct_uae_path"] = None

    # Load data
    data = load_data()

    # Create splits based on experiment type
    # Tag suffix for ablation
    uae_suffix = "_nouae" if no_uae else ""

    if exp == "p0":
        # Leave-one-G6-host-out
        assert fold in G6_HOSTS, f"fold must be one of {G6_HOSTS}"
        tag = f"loho_{fold}_s{seed}{uae_suffix}"
        train_data, val_data, test_data = make_loho_split(
            data, holdout_hosts=[fold], seed=seed
        )
        print(f"  P0: Hold out host '{fold}'")

    elif exp == "p1":
        # Block-out: G6 × 3d-TM
        tag = f"block_g6x3d_s{seed}{uae_suffix}"
        train_data, val_data, test_data = make_loho_split(
            data, block_hosts=set(G6_HOSTS), block_dopants=set(TM_3D), seed=seed
        )
        print(f"  P1: Hold out G6-TMD × 3d-TM block")

    else:
        raise ValueError(f"Unknown experiment: {exp}")

    print(f"  Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Analyze test set
    test_hosts = set(s["metadata"]["host"] for s in test_data)
    test_dopants = set(s["metadata"]["dopant"] for s in test_data)
    test_efs = [s["target"] for s in test_data]
    print(f"  Test: {len(test_hosts)} hosts, {len(test_dopants)} dopants, "
          f"Ef range [{min(test_efs):.1f}, {max(test_efs):.1f}]")

    # Create datasets and loaders
    train_ds = LOHODataset(train_data)
    val_ds = LOHODataset(val_data)
    test_ds = LOHODataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Train
    model, best_val, wall, history = train_model(train_loader, val_loader, device, seed, tag,
                                                  model_kwargs=model_kwargs)

    # Evaluate on test set
    print(f"\n  Evaluating on OOD test set ({len(test_data)} samples)...")
    preds, targets, test_mae, test_rmse = evaluate(model, test_loader, device)
    print(f"  ** OOD Test MAE: {test_mae:.4f} eV, RMSE: {test_rmse:.4f} eV **")

    # Naive baseline: predict mean of training targets
    train_mean = np.mean([s["target"] for s in train_data])
    naive_mae = np.abs(targets - train_mean).mean()
    print(f"  Naive baseline (predict train mean): MAE = {naive_mae:.4f} eV")

    # Per-host breakdown (for P0, single host; for P1, multiple)
    per_host = {}
    for i, s in enumerate(test_data):
        h = s["metadata"]["host"]
        if h not in per_host:
            per_host[h] = {"preds": [], "targets": []}
        per_host[h]["preds"].append(preds[i])
        per_host[h]["targets"].append(targets[i])

    print(f"\n  Per-host breakdown:")
    for h in sorted(per_host.keys()):
        p = np.array(per_host[h]["preds"])
        t = np.array(per_host[h]["targets"])
        h_mae = np.abs(p - t).mean()
        print(f"    {h:<12}: MAE={h_mae:.4f} (n={len(p)})")

    # Save results
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": exp,
        "fold": fold,
        "seed": seed,
        "tag": tag,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "best_val_mae": float(best_val),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "naive_baseline_mae": float(naive_mae),
        "wall_min": float(wall),
        "per_host_mae": {h: float(np.abs(np.array(per_host[h]["preds"]) -
                                          np.array(per_host[h]["targets"])).mean())
                        for h in per_host},
        "model_kwargs": model_kwargs,
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "warmup": WARMUP_EPOCHS, "swa_start": SWA_START,
        },
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(out_dir / "predictions.npz", preds=preds, targets=targets)
    print(f"\n  Saved -> {out_dir}")
    return results


def launch_all_p0(gpus, seed):
    """Launch all P0 folds in parallel using subprocess."""
    import subprocess

    gpu_list = [int(g) for g in gpus.split(",")]
    n_gpus = len(gpu_list)
    print(f"Launching all P0 folds on GPUs: {gpu_list}")

    procs = []
    for i, host in enumerate(G6_HOSTS):
        gpu = gpu_list[i % n_gpus]
        cmd = (f"CUDA_VISIBLE_DEVICES={gpu} "
               f"python scripts/ood_loho_train.py --exp p0 --fold {host} "
               f"--gpu 0 --seed {seed}")
        print(f"  [{host}] GPU {gpu}: {cmd}")
        p = subprocess.Popen(cmd, shell=True, cwd=str(ROOT))
        procs.append((host, p))

    # Wait for all to finish
    for host, p in procs:
        ret = p.wait()
        status = "OK" if ret == 0 else f"FAILED (rc={ret})"
        print(f"  [{host}] {status}")


def main():
    parser = argparse.ArgumentParser(description="OOD Leave-Out Experiments")
    parser.add_argument("--exp", choices=["p0", "p1"], required=True,
                        help="Experiment type: p0=leave-one-G6-host, p1=G6x3d-block")
    parser.add_argument("--fold", default=None,
                        help="For P0: which host to hold out (e.g., MoS2)")
    parser.add_argument("--all", action="store_true",
                        help="Launch all P0 folds in parallel")
    parser.add_argument("--gpus", default="0,1,6",
                        help="Comma-separated GPU IDs for --all mode")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID for single fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-uae", action="store_true",
                        help="Ablation: disable ct-UAE pretrained embeddings")
    args = parser.parse_args()

    if args.all and args.exp == "p0":
        launch_all_p0(args.gpus, args.seed)
        return

    if args.exp == "p0" and args.fold is None:
        parser.error("P0 requires --fold (host name) or --all")

    fold = args.fold if args.fold else "g6x3d"
    run_single_fold(args.exp, fold, args.seed, args.gpu, no_uae=args.no_uae)


if __name__ == "__main__":
    main()
