"""Occlusion attribution: per-atom contribution to predicted formation energy.

For each atom $i$ of a defect supercell, we mask it out from the model
(``atom_mask[i] = False``) and re-evaluate. The change in predicted $E_f$,
$$\\Delta_i = \\hat{E}_f^{(\\text{full})} - \\hat{E}_f^{(\\text{mask } i)},$$
gives a directly interpretable per-atom contribution: positive $\\Delta_i$
means "atom $i$ contributed +Δ to the formation energy".

We then visualise this as a colour-coded scatter of atomic positions in the
2D plane, with the dopant atom highlighted, allowing the energy localisation
to be read off at a glance: the more localised the heat-map around the defect
site, the stronger the model's "local" reasoning; spillover to remote atoms
is direct evidence of the long-range coupling that motivated the DAST design.

We aggregate over 100 test samples to produce three numbers:

  * mean |Δ| at the defect atom vs the mean |Δ| over non-defect atoms
  * effective localisation radius: smallest r such that 80% of total |Δ|
    lies within r angstrom of the defect
  * Pearson correlation between |Δ_i| and 1 / dist_to_defect (a strong
    correlation supports the claim that the model's reasoning is anchored
    on the defect)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def predict(model, batch, normalizer_mean, normalizer_std):
    with torch.no_grad():
        y_norm = model(batch).item()
    return y_norm * normalizer_std + normalizer_mean


def occlusion_per_atom(model, sample, normalizer_mean, normalizer_std):
    """Returns array Δ[i] = full_pred - pred_with_atom_i_masked."""
    n = sample["num_atoms"]
    base_batch = collate_fn([sample])
    full_pred = predict(model, base_batch, normalizer_mean, normalizer_std)

    deltas = np.zeros(n)
    masked_preds = np.zeros(n)
    for i in range(n):
        # build a fresh batch with atom i removed from the attention mask
        b = collate_fn([sample])
        b["atom_mask"][0, i] = False
        # also kill its row/col contributions in dist_matrix is unnecessary
        # because masked_fill in attention will already drop it, and the
        # local-graph branch references all atoms but we keep its features
        # intact (occlusion = "model can no longer see this atom in attn /
        # pooling readout").
        masked_preds[i] = predict(model, b, normalizer_mean, normalizer_std)
        deltas[i] = full_pred - masked_preds[i]
    return full_pred, masked_preds, deltas


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    cleaned_path = ROOT / "data/processed/cleaned_dataset.pkl"
    safe_path = ROOT / cfg["data_path"]
    if safe_path.exists():
        ds = CrystalGraphDataset(safe_path)
    else:
        ds = CrystalGraphDataset(cleaned_path)
    _, _, test_set = make_splits(
        ds, cfg.get("train_ratio", 0.8), cfg.get("val_ratio", 0.1),
        cfg.get("seed", 42),
    )

    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt_path = ROOT / "results/baseline_h128_aug_long_safe/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nmean = ckpt["normalizer"]["mean"]
    nstd = ckpt["normalizer"]["std"]

    # ---------- single-sample localisation figure ----------
    pick = None
    for i in range(len(test_set)):
        s = test_set[i]
        if s["defect_mask"].sum().item() == 1 and 28 <= s["num_atoms"] <= 50:
            pick = i; break
    if pick is None:
        pick = 0
    sample = test_set[pick]
    n = sample["num_atoms"]
    defect_idx = int(sample["defect_mask"].argmax().item())
    print(f"sample idx {pick} | n={n} | defect_idx={defect_idx}")

    full_pred, masked_preds, deltas = occlusion_per_atom(model, sample, nmean, nstd)
    target = sample["target"].item()
    print(f"target={target:.3f}  full_pred={full_pred:.3f}  defect Δ={deltas[defect_idx]:+.3f}")

    pos = sample["positions"].numpy()
    z = sample["x"][:, 0].numpy()  # not actual atomic number, but feature[0]
    # to plot real element labels we need numbers
    # numbers is in raw sample, but test_set returns processed; recover from data
    numbers_arr = ds.data[test_set.indices[pick]]["numbers"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    ax = axes[0]
    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=-np.abs(deltas).max(), vmax=np.abs(deltas).max())
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=deltas, cmap=cmap, norm=norm, s=200, edgecolors="k")
    ax.scatter(pos[defect_idx, 0], pos[defect_idx, 1], facecolor="none",
               edgecolor="lime", s=500, lw=2.5, label="defect")
    for i, (x, y) in enumerate(pos[:, :2]):
        ax.text(x, y, str(numbers_arr[i]), fontsize=6, ha="center", va="center")
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    ax.set_title(f"Per-atom Δ = E_f^full − E_f^mask_i\n"
                 f"target {target:.3f} eV; full {full_pred:.3f} eV")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Δ (eV)")

    ax = axes[1]
    dists_to_defect = np.linalg.norm(pos - pos[defect_idx], axis=1)
    abs_dlt = np.abs(deltas)
    ax.scatter(dists_to_defect, abs_dlt, color="tab:blue")
    ax.scatter(dists_to_defect[defect_idx], abs_dlt[defect_idx], color="red",
               s=120, label=f"defect (Δ={deltas[defect_idx]:+.3f})")
    ax.set_xlabel("Distance from defect (Å)")
    ax.set_ylabel("|Δ_i| (eV)")
    ax.set_title("Energy contribution vs distance from defect")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig_occlusion_localisation.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"saved {out}")

    # ---------- aggregate over a sample of test set ----------
    aggregate_n = 100
    print(f"aggregating occlusion stats over {aggregate_n} samples...")
    defect_d, other_d = [], []
    pearsons_inv_dist = []
    radii_80 = []
    radii_80_excl_defect = []
    fraction_at_defect = []
    for i in range(aggregate_n):
        s = test_set[i]
        n_i = s["num_atoms"]
        if n_i < 6 or s["defect_mask"].sum().item() != 1:
            continue
        d_i = int(s["defect_mask"].argmax().item())
        _, _, dlt = occlusion_per_atom(model, s, nmean, nstd)
        adlt = np.abs(dlt)
        defect_d.append(adlt[d_i])
        other = adlt.copy()
        other[d_i] = np.nan
        other_d.append(np.nanmean(other))
        fraction_at_defect.append(float(adlt[d_i] / max(adlt.sum(), 1e-9)))
        pos_i = s["positions"].numpy()
        d_arr = np.linalg.norm(pos_i - pos_i[d_i], axis=1)
        d_arr_safe = np.maximum(d_arr, 0.1)
        if n_i > 3:
            pearsons_inv_dist.append(
                float(np.corrcoef(adlt, 1.0 / d_arr_safe)[0, 1])
            )
        # full radius containing 80% of total |Δ|
        order = np.argsort(d_arr)
        cum = np.cumsum(adlt[order]) / max(adlt.sum(), 1e-9)
        r80 = float(d_arr[order][np.searchsorted(cum, 0.8)]) if cum[-1] > 0 else float("nan")
        radii_80.append(r80)
        # Now exclude defect atom and find the radius containing 80% of the
        # *residual* |Δ|. This tells us how delocalised the residual signal is.
        adlt_excl = adlt.copy()
        adlt_excl[d_i] = 0.0
        if adlt_excl.sum() > 1e-9:
            order_excl = np.argsort(d_arr)
            cum_excl = np.cumsum(adlt_excl[order_excl]) / adlt_excl.sum()
            r80x = float(d_arr[order_excl][np.searchsorted(cum_excl, 0.8)])
            radii_80_excl_defect.append(r80x)

    stats = {
        "n_samples_used": int(len(defect_d)),
        "mean_abs_delta_at_defect": float(np.mean(defect_d)),
        "mean_abs_delta_at_other_atoms": float(np.mean(other_d)),
        "ratio_defect_over_other": float(np.mean(defect_d) / max(np.mean(other_d), 1e-9)),
        "fraction_of_total_attribution_at_defect_atom_mean": float(np.mean(fraction_at_defect)),
        "fraction_of_total_attribution_at_defect_atom_std": float(np.std(fraction_at_defect)),
        "pearson_abs_delta_vs_inv_dist_mean": float(np.nanmean(pearsons_inv_dist)),
        "pearson_abs_delta_vs_inv_dist_std": float(np.nanstd(pearsons_inv_dist)),
        "radius_for_80pct_mean": float(np.nanmean(radii_80)),
        "radius_for_80pct_std": float(np.nanstd(radii_80)),
        "radius_for_80pct_excl_defect_mean": float(np.nanmean(radii_80_excl_defect)),
        "radius_for_80pct_excl_defect_std": float(np.nanstd(radii_80_excl_defect)),
    }
    print(json.dumps(stats, indent=2))
    with open(ROOT / "results/occlusion_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("saved results/occlusion_stats.json")


if __name__ == "__main__":
    main()
