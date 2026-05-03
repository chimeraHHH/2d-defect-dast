"""Simulated active-learning loop: UQ-guided vs random sample selection.

Demonstrates that MC-Dropout uncertainty can guide iterative sample
selection to reduce test MAE faster than random selection -- the core
value proposition of UQ in a materials-science active-learning pipeline.

Simulation protocol
-------------------
1. Start with 10% of the training data as the initial labeled pool.
2. Train the model on the labeled pool.
3. Predict on the unlabeled pool using MC-Dropout (K stochastic forward
   passes) to obtain per-sample uncertainty (std of predictions).
4. **Active** strategy: add the K=50 highest-uncertainty samples.
   **Random** strategy: add 50 random samples (averaged over 3 seeds).
5. Repeat for 15 rounds, evaluating test MAE after each round.

To keep each round tractable we use differential learning rates:
a small LR for the frozen backbone and a higher LR for the readout head.
Each round trains for 20 epochs with early stopping on a held-out
validation set.

Outputs
-------
- ``paper/figures/fig_active_learning_loop.png``  -- learning curves
- ``results/active_learning_loop.json``            -- tabular results + AULC
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, split_indices  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

# ── paths ───────────────────────────────────────────────────────────────
CKPT_PATH = ROOT / "results" / "baseline_h128_aug_long_safe" / "best.pt"
DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-parameters ────────────────────────────────────────────────────
INITIAL_FRAC = 0.10          # fraction of training data to start with
N_ROUNDS = 15                # active-learning rounds
K_PER_ROUND = 50             # samples added per round
MC_K = 10                    # MC-Dropout forward passes
EPOCHS_PER_ROUND = 20        # fine-tuning epochs per round
BATCH_SIZE = 16
LR_BACKBONE = 1e-5           # low LR for frozen backbone
LR_READOUT = 1e-3            # higher LR for readout head
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
N_RANDOM_SEEDS = 3           # random-baseline averaging seeds
GLOBAL_SEED = 42
PATIENCE = 5                 # early stopping patience


# ── helpers ─────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Normalizer:
    """Target normalizer recomputed at each round from the labeled set."""

    def __init__(self, values: torch.Tensor) -> None:
        self.mean = float(values.mean().item())
        self.std = float(values.std().item()) + 1e-6

    def norm(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def denorm(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.std + self.mean


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ── model utilities ─────────────────────────────────────────────────────
def load_pretrained_model(device: torch.device) -> Tuple[CrystalTransformer, dict]:
    """Load the pretrained CrystalTransformer and checkpoint normalizer."""
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_kwargs = cfg.get("model_kwargs", {})
    model = CrystalTransformer(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt["normalizer"]


def make_fresh_model(device: torch.device) -> CrystalTransformer:
    """Return a fresh copy of the pretrained model (weights re-loaded)."""
    model, _ = load_pretrained_model(device)
    return model


def build_optimizer(model: CrystalTransformer) -> torch.optim.Optimizer:
    """Differential LR: low for backbone, high for readout."""
    readout_params = list(model.readout.parameters())
    readout_ids = {id(p) for p in readout_params}
    backbone_params = [p for p in model.parameters() if id(p) not in readout_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": readout_params, "lr": LR_READOUT},
    ], weight_decay=WEIGHT_DECAY)


# ── training / evaluation ──────────────────────────────────────────────
def train_one_round(
    model: CrystalTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    epochs: int = EPOCHS_PER_ROUND,
) -> CrystalTransformer:
    """Fine-tune the model on the current labeled set with early stopping."""
    optimizer = build_optimizer(model)
    criterion = nn.MSELoss()
    best_val_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            target_norm = normalizer.norm(batch["target"])
            pred_norm = model(batch)
            loss = criterion(pred_norm, target_norm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # validation
        val_mae = evaluate_mae(model, val_loader, normalizer, device)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_mae(
    model: CrystalTransformer,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> float:
    """Compute MAE in original (eV) units."""
    model.eval()
    abs_err, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred_norm = model(batch)
            pred = normalizer.denorm(pred_norm)
            abs_err += (pred - batch["target"]).abs().sum().item()
            n += batch["target"].numel()
    return abs_err / max(n, 1)


def mc_dropout_predict(
    model: CrystalTransformer,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    k: int = MC_K,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run K stochastic forward passes (dropout enabled) and return (mean, std).

    The CrystalTransformer uses LayerNorm + Dropout (no BatchNorm), so
    calling model.train() safely activates dropout without side-effects on
    normalisation statistics.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    all_preds: List[np.ndarray] = []
    for _ in range(k):
        preds_run: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                pred_norm = model(batch)
                pred = normalizer.denorm(pred_norm)
                preds_run.append(pred.cpu().numpy())
        all_preds.append(np.concatenate(preds_run))

    # re-enable grads for subsequent training
    for p in model.parameters():
        p.requires_grad_(True)

    P = np.stack(all_preds)  # (K, N)
    return P.mean(0), P.std(0, ddof=1)


# ── data helpers ────────────────────────────────────────────────────────
def make_normalizer(dataset: CrystalGraphDataset, indices: List[int]) -> Normalizer:
    """Build a Normalizer from the targets of `indices` into `dataset`."""
    targets = torch.tensor(
        [dataset.data[i]["target"] for i in indices], dtype=torch.float32
    )
    return Normalizer(targets)


def make_loader(dataset: CrystalGraphDataset, indices: List[int],
                shuffle: bool = False) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      collate_fn=collate_fn)


# ── single active-learning trajectory ──────────────────────────────────
def run_trajectory(
    dataset: CrystalGraphDataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    strategy: str,
    seed: int,
    device: torch.device,
) -> List[dict]:
    """Run one full active-learning trajectory.

    Parameters
    ----------
    strategy : ``"active"`` (MC-Dropout UQ) or ``"random"``.
    seed : controls initial labeled-set sampling and random selection.

    Returns
    -------
    List of per-round dicts with ``n_labeled`` and ``test_mae``.
    """
    rng = np.random.default_rng(seed)

    # initial split: 10% labeled, 90% pool
    n_train = len(train_indices)
    n_init = max(1, int(INITIAL_FRAC * n_train))
    perm = rng.permutation(n_train)
    labeled_mask = np.zeros(n_train, dtype=bool)
    labeled_mask[perm[:n_init]] = True

    val_loader = make_loader(dataset, val_indices, shuffle=False)
    test_loader = make_loader(dataset, test_indices, shuffle=False)

    history: List[dict] = []
    train_arr = np.array(train_indices)

    for rnd in range(N_ROUNDS + 1):  # round 0 = initial training
        labeled_idx = train_arr[labeled_mask].tolist()
        pool_idx = train_arr[~labeled_mask].tolist()

        # normalizer from current labeled set
        normalizer = make_normalizer(dataset, labeled_idx)

        # fresh model each round (reload pretrained weights)
        model = make_fresh_model(device)

        # fine-tune on labeled set
        train_loader = make_loader(dataset, labeled_idx, shuffle=True)
        model = train_one_round(model, train_loader, val_loader, normalizer, device)

        # evaluate on test
        test_mae = evaluate_mae(model, test_loader, normalizer, device)
        history.append({"round": rnd, "n_labeled": len(labeled_idx),
                        "test_mae": test_mae})
        print(f"  [{strategy} seed={seed}] round {rnd:2d} | "
              f"labeled {len(labeled_idx):4d} | test MAE {test_mae:.4f}")

        if rnd == N_ROUNDS or len(pool_idx) == 0:
            break

        # select next batch
        k = min(K_PER_ROUND, len(pool_idx))

        if strategy == "active":
            pool_loader = make_loader(dataset, pool_idx, shuffle=False)
            _, sigma = mc_dropout_predict(model, pool_loader, normalizer, device)
            top_k_local = np.argsort(sigma)[-k:]  # highest uncertainty
            # map local pool indices back to train_arr mask positions
            pool_positions = np.where(~labeled_mask)[0]
            selected_positions = pool_positions[top_k_local]
        else:
            pool_positions = np.where(~labeled_mask)[0]
            selected_positions = rng.choice(pool_positions, size=k, replace=False)

        labeled_mask[selected_positions] = True

    return history


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.time()
    device = get_device()
    print(f"Device: {device}")

    set_seed(GLOBAL_SEED)

    # load dataset and splits
    dataset = CrystalGraphDataset(DATA_PATH)
    train_idx, val_idx, test_idx = split_indices(len(dataset), 0.8, 0.1, 42)
    print(f"Dataset: {len(dataset)} samples | "
          f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

    # ── active trajectory ───────────────────────────────────────────────
    print("\n=== Active (UQ-guided) trajectory ===")
    active_history = run_trajectory(
        dataset, train_idx, val_idx, test_idx,
        strategy="active", seed=GLOBAL_SEED, device=device,
    )

    # ── random trajectories (averaged over N_RANDOM_SEEDS) ──────────────
    random_histories: List[List[dict]] = []
    for rs in range(N_RANDOM_SEEDS):
        seed_i = GLOBAL_SEED + rs + 100
        print(f"\n=== Random trajectory (seed {seed_i}) ===")
        rh = run_trajectory(
            dataset, train_idx, val_idx, test_idx,
            strategy="random", seed=seed_i, device=device,
        )
        random_histories.append(rh)

    # ── aggregate random runs ───────────────────────────────────────────
    n_rounds_actual = min(len(h) for h in random_histories)
    rand_n_labeled = [random_histories[0][i]["n_labeled"] for i in range(n_rounds_actual)]
    rand_maes = np.array([
        [h[i]["test_mae"] for i in range(n_rounds_actual)]
        for h in random_histories
    ])
    rand_mae_mean = rand_maes.mean(axis=0).tolist()
    rand_mae_std = rand_maes.std(axis=0).tolist()

    active_n_labeled = [h["n_labeled"] for h in active_history]
    active_maes = [h["test_mae"] for h in active_history]

    # ── AULC (area under learning curve) ────────────────────────────────
    def aulc(x: List[float], y: List[float]) -> float:
        """Trapezoidal area under (n_labeled, MAE) curve."""
        return float(np.trapz(y, x))

    active_aulc = aulc(active_n_labeled, active_maes)
    random_aulc = aulc(rand_n_labeled, rand_mae_mean)
    aulc_reduction = (random_aulc - active_aulc) / random_aulc * 100

    print(f"\n{'='*60}")
    print(f"AULC  active:  {active_aulc:.2f}")
    print(f"AULC  random:  {random_aulc:.2f}")
    print(f"AULC  reduction: {aulc_reduction:.1f}%")
    print(f"{'='*60}")

    # ── save JSON results ───────────────────────────────────────────────
    results = {
        "config": {
            "initial_frac": INITIAL_FRAC,
            "n_rounds": N_ROUNDS,
            "k_per_round": K_PER_ROUND,
            "mc_k": MC_K,
            "epochs_per_round": EPOCHS_PER_ROUND,
            "batch_size": BATCH_SIZE,
            "lr_backbone": LR_BACKBONE,
            "lr_readout": LR_READOUT,
            "n_random_seeds": N_RANDOM_SEEDS,
            "global_seed": GLOBAL_SEED,
        },
        "active": {
            "n_labeled": active_n_labeled,
            "test_mae": active_maes,
            "aulc": active_aulc,
        },
        "random": {
            "n_labeled": rand_n_labeled,
            "test_mae_mean": rand_mae_mean,
            "test_mae_std": rand_mae_std,
            "aulc": random_aulc,
        },
        "aulc_reduction_pct": aulc_reduction,
        "wall_time_sec": time.time() - t_start,
    }
    out_json = RESULTS_DIR / "active_learning_loop.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_json}")

    # ── figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(active_n_labeled, active_maes, "o-", color="#1f77b4",
            linewidth=2, markersize=5, label="Active (MC-Dropout UQ)")

    ax.plot(rand_n_labeled, rand_mae_mean, "s--", color="#d62728",
            linewidth=2, markersize=5, label=f"Random (avg of {N_RANDOM_SEEDS} seeds)")
    ax.fill_between(
        rand_n_labeled,
        [m - s for m, s in zip(rand_mae_mean, rand_mae_std)],
        [m + s for m, s in zip(rand_mae_mean, rand_mae_std)],
        alpha=0.18, color="#d62728",
    )

    ax.set_xlabel("Number of labeled training samples", fontsize=12)
    ax.set_ylabel("Test MAE (eV)", fontsize=12)
    ax.set_title("Simulated Active Learning: UQ-guided vs Random Selection",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    # annotate AULC
    ax.text(
        0.02, 0.02,
        f"AULC reduction: {aulc_reduction:.1f}%\n"
        f"(active {active_aulc:.1f} vs random {random_aulc:.1f})",
        transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    out_fig = FIG_DIR / "fig_active_learning_loop.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)
    print(f"figure saved -> {out_fig}")

    elapsed = time.time() - t_start
    print(f"\ntotal wall time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
