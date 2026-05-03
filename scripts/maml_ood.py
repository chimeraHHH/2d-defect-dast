"""MAML-style meta-learning for OOD chemical-family adaptation.

For each LOHO host, we simulate a few-shot OOD scenario:
  1. Load the pre-trained CrystalTransformer checkpoint.
  2. Split the held-out host samples into k *support* and the rest *query*.
  3. Compare three strategies on the query set:
       (a) Zero-shot   -- use the pre-trained model directly.
       (b) Naive FT    -- copy the model, fine-tune on k support samples for
                          N_inner SGD steps with inner_lr, evaluate.
       (c) FOMAML      -- first-order MAML: same inner-loop as naive FT, but
                          we use a Reptile-style parameter update (which, for a
                          single task, reduces to vanilla SGD on the support
                          set).  The conceptual distinction is that FOMAML
                          re-initialises from the *meta-parameters* at every
                          new task, whereas naive FT simply continues from the
                          pre-trained weights with standard SGD -- here we
                          implement the inner loop with a *higher inner_lr*
                          and only adapt the readout head (keeping the backbone
                          frozen) to test whether rapid head-only adaptation is
                          more effective than full-model fine-tuning.

Since we are operating in a single-task-per-host setting (no outer-loop meta-
optimisation across hosts), the FOMAML column effectively tests *selective
fine-tuning* (readout-only) vs *full fine-tuning* (naive FT) vs *no adaptation*
(zero-shot).  This is the practically relevant comparison for the LOHO OOD
scenario.

We sweep over:
  k       = {5, 10, 20}    (support-set size)
  N_inner = {3, 5, 10}     (inner-loop gradient steps)

Each (host, k, N_inner) trial is repeated over 3 random support/query splits
(seeds 0, 1, 2) and averaged.

Outputs:
  results/maml_ood.json               -- full numeric results
  paper/figures/fig_maml_ood_bars.png  -- grouped bar chart
"""
from __future__ import annotations

import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

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

from src.dataset import CrystalGraphDataset, collate_fn  # noqa: E402
from src.models import CrystalTransformer                 # noqa: E402
from src.train import Normalizer, move_batch               # noqa: E402

# ------------------------------------------------------------------ constants
HOSTS = ["MoS2", "Cr2I6", "C2H2", "TaSe2", "MoSSe"]
K_VALUES = [5, 10, 20]
N_INNER_VALUES = [3, 5, 10]
INNER_LR = 1e-3
INNER_LR_MAML = 3e-3       # higher LR for head-only MAML adaptation
N_SEEDS = 3                 # random support/query splits per trial
BATCH_SIZE = 32
CKPT_RUN = "baseline_h128_aug_long_safe"

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "paper" / "figures"


# ------------------------------------------------------------------ helpers
def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_pretrained(device: torch.device):
    """Return (model, normalizer) from the pre-trained checkpoint."""
    ckpt_path = RESULTS_DIR / CKPT_RUN / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_kwargs = cfg.get("model_kwargs", {
        "atom_fea_len": 9,
        "hidden_dim": 128,
        "n_local_layers": 3,
        "n_global_layers": 2,
        "num_heads": 4,
    })
    model = CrystalTransformer(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    norm = Normalizer(torch.zeros(2))
    if "normalizer" in ckpt:
        norm.mean = float(ckpt["normalizer"]["mean"])
        norm.std = float(ckpt["normalizer"]["std"])
    else:
        raise RuntimeError("Checkpoint does not contain normalizer state")
    return model, norm


def _get_host_indices(dataset: CrystalGraphDataset, host: str):
    """Return list of dataset indices whose host matches *host*."""
    indices = []
    for i, sample in enumerate(dataset.data):
        h = sample.get("metadata", {}).get("host", "")
        if h == host:
            indices.append(i)
    return indices


def _split_support_query(indices, k, seed):
    """Randomly pick k support indices; the rest become query."""
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    support = shuffled[:k]
    query = shuffled[k:]
    return support, query


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate_mae(model, loader, normalizer, device):
    """Return MAE (eV) on the given loader."""
    model.eval()
    abs_err, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        preds_norm = model(batch)
        preds = normalizer.denorm(preds_norm)
        abs_err += (preds - batch["target"]).abs().sum().item()
        n += batch["target"].numel()
    return abs_err / max(n, 1)


# ------------------------------------------------------------------ adapt
def naive_finetune(base_model, support_loader, normalizer, device,
                   n_steps, lr):
    """Full-model fine-tuning on the support set for n_steps steps.

    Returns a *new* model (does not mutate base_model).
    """
    model = copy.deepcopy(base_model)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    step = 0
    while step < n_steps:
        for batch in support_loader:
            if step >= n_steps:
                break
            batch = move_batch(batch, device)
            target_norm = normalizer.norm(batch["target"])
            preds_norm = model(batch)
            loss = criterion(preds_norm, target_norm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
    return model


def maml_adapt(base_model, support_loader, normalizer, device,
               n_steps, lr):
    """First-order MAML / head-only adaptation on the support set.

    Only the readout head parameters receive gradients; the backbone (embed,
    local_layers, global_layers) is frozen.  This simulates a MAML inner loop
    where rapid adaptation of a small parameter subset is preferred.

    Returns a *new* model.
    """
    model = copy.deepcopy(base_model)
    # Freeze backbone, unfreeze readout
    for name, param in model.named_parameters():
        if "readout" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.train()

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    step = 0
    while step < n_steps:
        for batch in support_loader:
            if step >= n_steps:
                break
            batch = move_batch(batch, device)
            target_norm = normalizer.norm(batch["target"])
            preds_norm = model(batch)
            loss = criterion(preds_norm, target_norm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1

    # unfreeze everything for later use (evaluation doesn't need grad, but
    # keeps the object in a clean state)
    for param in model.parameters():
        param.requires_grad = True
    return model


# ------------------------------------------------------------------ figure
def make_figure(results, hosts, k_values, fig_path):
    """Grouped bar chart: for each host, bars for zero-shot / naive FT / MAML.

    Uses k=10, N_inner=5 as the "headline" comparison; the full sweep is in
    the JSON output.
    """
    headline_k = 10
    headline_n = 5

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(hosts))
    width = 0.22

    zero_vals = []
    naive_vals = []
    naive_errs = []
    maml_vals = []
    maml_errs = []

    for host in hosts:
        key = f"{host}_k{headline_k}_n{headline_n}"
        entry = results.get(key)
        if entry is None:
            # fall back to whichever k/n is available
            for k in k_values:
                for n in N_INNER_VALUES:
                    alt = f"{host}_k{k}_n{n}"
                    if alt in results:
                        entry = results[alt]
                        break
                if entry is not None:
                    break
        if entry is None:
            zero_vals.append(0)
            naive_vals.append(0)
            naive_errs.append(0)
            maml_vals.append(0)
            maml_errs.append(0)
            continue
        zero_vals.append(entry["zero_shot_mae"])
        naive_vals.append(entry["naive_ft_mae_mean"])
        naive_errs.append(entry["naive_ft_mae_std"])
        maml_vals.append(entry["maml_mae_mean"])
        maml_errs.append(entry["maml_mae_std"])

    bars_z = ax.bar(x - width, zero_vals, width, label="Zero-shot",
                    color="tab:gray", edgecolor="black", linewidth=0.5)
    bars_n = ax.bar(x, naive_vals, width, yerr=naive_errs,
                    label="Naive FT (full model)", color="tab:blue",
                    edgecolor="black", linewidth=0.5, capsize=3)
    bars_m = ax.bar(x + width, maml_vals, width, yerr=maml_errs,
                    label="FOMAML (head-only)", color="tab:red",
                    edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_ylabel("Test MAE (eV)")
    ax.set_title(f"MAML-style OOD Adaptation  (k={headline_k}, N_inner={headline_n})")
    ax.set_xticks(x)
    ax.set_xticklabels(hosts, rotation=25, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {fig_path}")


# ------------------------------------------------------------------ main
def main():
    device = _detect_device()
    print(f"Device: {device}")
    print(f"Loading pre-trained checkpoint from {CKPT_RUN} ...")
    base_model, normalizer = _load_pretrained(device)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"Model params: {n_params / 1e6:.3f}M")

    # Load the full cleaned dataset
    ds_path = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
    print(f"Loading dataset from {ds_path} ...")
    dataset = CrystalGraphDataset(ds_path)
    print(f"Dataset size: {len(dataset)}")

    all_results: dict = {}
    summary_rows = []

    for host in HOSTS:
        host_indices = _get_host_indices(dataset, host)
        n_host = len(host_indices)
        if n_host == 0:
            print(f"\n[WARNING] Host '{host}' has 0 samples -- skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Host: {host}  ({n_host} samples)")
        print(f"{'='*60}")

        for k in K_VALUES:
            if k >= n_host:
                print(f"  k={k} >= n_host={n_host}, skipping.")
                continue
            for n_inner in N_INNER_VALUES:
                tag = f"{host}_k{k}_n{n_inner}"
                print(f"\n  --- k={k}, N_inner={n_inner} ---")

                zero_maes = []
                naive_maes = []
                maml_maes = []

                for seed in range(N_SEEDS):
                    support_idx, query_idx = _split_support_query(
                        host_indices, k, seed=seed
                    )
                    if len(query_idx) == 0:
                        print(f"    seed={seed}: no query samples, skipping")
                        continue

                    support_loader = DataLoader(
                        Subset(dataset, support_idx),
                        batch_size=min(BATCH_SIZE, k),
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    query_loader = DataLoader(
                        Subset(dataset, query_idx),
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )

                    # (a) zero-shot
                    base_model.eval()
                    zs_mae = evaluate_mae(base_model, query_loader, normalizer, device)
                    zero_maes.append(zs_mae)

                    # (b) naive fine-tune (full model)
                    ft_model = naive_finetune(
                        base_model, support_loader, normalizer, device,
                        n_steps=n_inner, lr=INNER_LR,
                    )
                    nft_mae = evaluate_mae(ft_model, query_loader, normalizer, device)
                    naive_maes.append(nft_mae)
                    del ft_model

                    # (c) FOMAML (head-only adaptation)
                    maml_model = maml_adapt(
                        base_model, support_loader, normalizer, device,
                        n_steps=n_inner, lr=INNER_LR_MAML,
                    )
                    m_mae = evaluate_mae(maml_model, query_loader, normalizer, device)
                    maml_maes.append(m_mae)
                    del maml_model

                    print(
                        f"    seed={seed}: zero={zs_mae:.3f}  "
                        f"naive_ft={nft_mae:.3f}  maml={m_mae:.3f}"
                    )

                if not zero_maes:
                    continue

                entry = {
                    "host": host,
                    "k": k,
                    "n_inner": n_inner,
                    "n_host_samples": n_host,
                    "n_query": n_host - k,
                    "zero_shot_mae": float(np.mean(zero_maes)),
                    "zero_shot_mae_std": float(np.std(zero_maes)),
                    "naive_ft_mae_mean": float(np.mean(naive_maes)),
                    "naive_ft_mae_std": float(np.std(naive_maes)),
                    "maml_mae_mean": float(np.mean(maml_maes)),
                    "maml_mae_std": float(np.std(maml_maes)),
                    "naive_ft_maes": [float(v) for v in naive_maes],
                    "maml_maes": [float(v) for v in maml_maes],
                    "zero_shot_maes": [float(v) for v in zero_maes],
                }
                # Relative improvement vs zero-shot
                zs = entry["zero_shot_mae"]
                entry["naive_ft_rel_improvement_pct"] = (
                    (zs - entry["naive_ft_mae_mean"]) / zs * 100 if zs > 0 else 0
                )
                entry["maml_rel_improvement_pct"] = (
                    (zs - entry["maml_mae_mean"]) / zs * 100 if zs > 0 else 0
                )

                all_results[tag] = entry
                summary_rows.append(entry)
                print(
                    f"    AVG: zero={entry['zero_shot_mae']:.3f}  "
                    f"naive_ft={entry['naive_ft_mae_mean']:.3f} ({entry['naive_ft_rel_improvement_pct']:+.1f}%)  "
                    f"maml={entry['maml_mae_mean']:.3f} ({entry['maml_rel_improvement_pct']:+.1f}%)"
                )

    # ---------------------------------------------------------------- save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "maml_ood.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {out_json}")

    # ---------------------------------------------------------------- figure
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "fig_maml_ood_bars.png"
    make_figure(all_results, HOSTS, K_VALUES, fig_path)

    # ---------------------------------------------------------------- table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE  (averaged over seeds)")
    print(f"{'='*70}")
    fmt = "{:<8} {:>4} {:>4} {:>10} {:>10} {:>10} {:>8} {:>8}"
    print(fmt.format("Host", "k", "N", "Zero-shot", "Naive FT", "FOMAML",
                      "FT imp%", "MAML imp%"))
    print("-" * 70)
    for r in summary_rows:
        print(fmt.format(
            r["host"], r["k"], r["n_inner"],
            f"{r['zero_shot_mae']:.3f}",
            f"{r['naive_ft_mae_mean']:.3f}",
            f"{r['maml_mae_mean']:.3f}",
            f"{r['naive_ft_rel_improvement_pct']:+.1f}",
            f"{r['maml_rel_improvement_pct']:+.1f}",
        ))

    # Best config per host
    print(f"\n{'='*70}")
    print("BEST CONFIG PER HOST")
    print(f"{'='*70}")
    by_host: dict = defaultdict(list)
    for r in summary_rows:
        by_host[r["host"]].append(r)
    for host in HOSTS:
        if host not in by_host:
            continue
        entries = by_host[host]
        # pick the config with lowest MAML MAE
        best = min(entries, key=lambda e: e["maml_mae_mean"])
        print(
            f"  {host:8s}: k={best['k']}, N={best['n_inner']}  "
            f"MAML={best['maml_mae_mean']:.3f} eV  "
            f"(vs zero-shot {best['zero_shot_mae']:.3f}, "
            f"improvement {best['maml_rel_improvement_pct']:+.1f}%)"
        )


if __name__ == "__main__":
    main()
