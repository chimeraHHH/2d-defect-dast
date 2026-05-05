"""Phase A4 — correlate per-atom attribution against per-atom physical
descriptors.

Loads:
  results/phase_a_descriptors.npz         (per-atom physics features)
  results/phase_a_occlusion_per_atom.npz  (per-atom |Δ_i|)

For each per-atom physical feature compute:
  Spearman ρ between |Δ_i| and the feature
  Pearson r same
  Compared against:  Spearman ρ vs (1 / dist_to_defect)  — the trivial
    distance-only baseline that the v1.2 paper already reported.

The headline question is whether the GNN's per-atom attribution
correlates more strongly with PHYSICAL strain / coordination change
than with bare distance, which would convert the v1.2 "9 Å attribution
radius" claim into a physical-strain-field claim rather than a pure
attribution-space artefact.

We additionally compute these correlations within each shell
(0-3, 3-5, 5-7, 7-9, >9 Å from defect) to surface scale-dependent
patterns.

Output: results/phase_a_correlations.json + a figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr, pearsonr  # noqa: E402

DESC = ROOT / "results" / "phase_a_descriptors.npz"
OCC = ROOT / "results" / "phase_a_occlusion_per_atom.npz"


def main():
    d = np.load(DESC, allow_pickle=True)
    o = np.load(OCC)

    # build per-atom delta array aligned to the 1065-test descriptor flat
    # array (occlusion only covered 590 samples; descriptor covered 1065).
    # We need to align by sample_id.
    occ_sids = o["sample_id"]
    occ_offsets = o["sample_offsets"]
    occ_delta = o["delta_per_atom"]

    desc_sids = d["sample_id"]   # one entry per atom — repeated within each sample
    desc_offsets = d["sample_offsets"]

    # The desc data also has sample_offsets and per-atom arrays. Both NPZs
    # share the same per-atom layout per sample (same atom ordering).
    # However occlusion only has the SUBSET of samples that fit (n_atoms<=100, defect_mask present).
    # We need to find which sample IDs are common.
    desc_sids_per_sample = []
    n_samples_desc = len(desc_offsets) - 1
    for i in range(n_samples_desc):
        a, b = desc_offsets[i], desc_offsets[i + 1]
        if a < b:
            desc_sids_per_sample.append(int(d["sample_id"][a]))
        else:
            desc_sids_per_sample.append(-1)
    desc_sids_per_sample = np.array(desc_sids_per_sample)

    # pair occ samples with desc samples
    occ_set = set(occ_sids.tolist())
    common = []  # list of (occ_sample_idx, desc_sample_idx)
    for j, sid in enumerate(occ_sids):
        try:
            di = int(np.where(desc_sids_per_sample == int(sid))[0][0])
            common.append((j, di))
        except IndexError:
            continue
    print(f"common samples: {len(common)} / {len(occ_sids)} (occ) and "
          f"{n_samples_desc} (desc)")

    # Concatenate per-atom delta + per-atom physics aligned
    delta_flat = []
    distance_flat = []
    bond_strain_max_flat = []
    bond_strain_mean_flat = []
    coord_change_flat = []
    shell_idx_flat = []
    is_defect_flat = []
    for occ_j, desc_i in common:
        a_o, b_o = occ_offsets[occ_j], occ_offsets[occ_j + 1]
        a_d, b_d = desc_offsets[desc_i], desc_offsets[desc_i + 1]
        if (b_o - a_o) != (b_d - a_d):
            # different n_atoms (shouldn't happen but skip)
            continue
        delta_flat.append(occ_delta[a_o:b_o])
        distance_flat.append(d["distance_to_defect"][a_d:b_d])
        bond_strain_max_flat.append(d["bond_strain_max"][a_d:b_d])
        bond_strain_mean_flat.append(d["bond_strain_mean"][a_d:b_d])
        coord_change_flat.append(d["coord_change"][a_d:b_d])
        shell_idx_flat.append(d["shell_index"][a_d:b_d])
        is_defect_flat.append(d["is_defect"][a_d:b_d])

    delta_flat = np.concatenate(delta_flat)
    distance_flat = np.concatenate(distance_flat)
    bond_strain_max_flat = np.concatenate(bond_strain_max_flat)
    bond_strain_mean_flat = np.concatenate(bond_strain_mean_flat)
    coord_change_flat = np.concatenate(coord_change_flat)
    shell_idx_flat = np.concatenate(shell_idx_flat)
    is_defect_flat = np.concatenate(is_defect_flat)
    print(f"total atoms aligned: {delta_flat.size}")

    # Define inverse distance (the v1.2 baseline correlation)
    inv_dist = 1.0 / np.maximum(distance_flat, 0.5)

    def _corr(x, y, label):
        m = np.isfinite(x) & np.isfinite(y) & (np.std(x) > 0) & (np.std(y) > 0)
        if m.sum() < 10:
            return {"label": label, "n": int(m.sum()), "spearman": None, "pearson": None}
        sp = spearmanr(x[m], y[m]).statistic
        pe = pearsonr(x[m], y[m]).statistic
        return {"label": label, "n": int(m.sum()),
                "spearman": float(sp), "pearson": float(pe)}

    # Aggregate over ALL atoms
    print("\n=== overall correlations (all atoms) ===")
    overall = []
    for x, label in [
        (inv_dist, "1/dist_to_defect"),
        (-distance_flat, "-distance_to_defect"),
        (bond_strain_max_flat, "bond_strain_max"),
        (bond_strain_mean_flat, "bond_strain_mean"),
        (coord_change_flat.astype(float), "coord_change"),
    ]:
        r = _corr(x, delta_flat, label)
        overall.append(r)
        print(f"  |Δ| vs {label:<28s}  ρ={r['spearman']:+.3f}  Pearson={r['pearson']:+.3f}  n={r['n']}")

    # Excluding defect atom (same as v1.2 protocol — the defect itself
    # dominates so we want the "remaining attribution field")
    nondef_mask = is_defect_flat == 0
    print("\n=== excluding defect atom ===")
    nondef = []
    for x, label in [
        (inv_dist, "1/dist_to_defect"),
        (-distance_flat, "-distance_to_defect"),
        (bond_strain_max_flat, "bond_strain_max"),
        (bond_strain_mean_flat, "bond_strain_mean"),
        (coord_change_flat.astype(float), "coord_change"),
    ]:
        x_m = x[nondef_mask]
        y_m = delta_flat[nondef_mask]
        r = _corr(x_m, y_m, label)
        nondef.append(r)
        print(f"  |Δ| vs {label:<28s}  ρ={r['spearman']:+.3f}  Pearson={r['pearson']:+.3f}  n={r['n']}")

    # Per-shell correlation (excluding defect)
    print("\n=== per-shell correlation (|Δ| vs bond_strain_max, excluding defect) ===")
    per_shell = []
    for k, name in enumerate(["0-3 Å", "3-5 Å", "5-7 Å", "7-9 Å", ">9 Å"]):
        m = nondef_mask & (shell_idx_flat == k)
        if m.sum() < 20:
            print(f"  shell {name}: n={m.sum()} (skipped)")
            continue
        r_strain = _corr(bond_strain_max_flat[m], delta_flat[m],
                          f"shell {name} bond_strain_max")
        r_dist = _corr(inv_dist[m], delta_flat[m],
                        f"shell {name} 1/distance")
        per_shell.append({
            "shell": name,
            "n": int(m.sum()),
            "spearman_bond_strain_max": r_strain["spearman"],
            "spearman_inv_dist": r_dist["spearman"],
            "pearson_bond_strain_max": r_strain["pearson"],
            "pearson_inv_dist": r_dist["pearson"],
        })
        print(f"  {name:<8s}  n={m.sum():>5d}  "
              f"|Δ| vs bond_strain_max ρ={r_strain['spearman']:+.3f}  "
              f"|Δ| vs 1/dist ρ={r_dist['spearman']:+.3f}")

    # Save & make figure
    out = {
        "n_atoms_aligned": int(delta_flat.size),
        "n_atoms_excluding_defect": int(nondef_mask.sum()),
        "overall_correlations": overall,
        "excluding_defect_correlations": nondef,
        "per_shell_correlations": per_shell,
    }
    out_path = ROOT / "results" / "phase_a_correlations.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")

    # Three-panel figure: |Δ| vs (1) distance (2) bond_strain (3) shell breakdown
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    nondef_d = delta_flat[nondef_mask]
    nondef_dist = distance_flat[nondef_mask]
    nondef_strain = bond_strain_max_flat[nondef_mask]

    ax = axes[0]
    ax.scatter(nondef_dist, nondef_d, s=3, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Distance to defect (Å)")
    ax.set_ylabel("|Δ_i| (eV)")
    ax.set_title(f"|Δ| vs distance (non-defect)\nρ = {nondef[1]['spearman']:+.3f}")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.scatter(nondef_strain, nondef_d, s=3, alpha=0.15, color="#9467bd")
    ax.set_xlabel("Bond strain max (relative)")
    ax.set_ylabel("|Δ_i| (eV)")
    ax.set_title(f"|Δ| vs bond_strain (non-defect)\nρ = {nondef[2]['spearman']:+.3f}")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    if per_shell:
        shells = [r["shell"] for r in per_shell]
        rho_strain = [r["spearman_bond_strain_max"] for r in per_shell]
        rho_dist = [r["spearman_inv_dist"] for r in per_shell]
        x_pos = np.arange(len(shells))
        w = 0.35
        ax.bar(x_pos - w / 2, rho_strain, w, label="ρ vs bond_strain_max",
               color="#9467bd")
        ax.bar(x_pos + w / 2, rho_dist, w, label="ρ vs 1/distance",
               color="#1f77b4")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(shells)
        ax.set_xlabel("Shell from defect")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Per-shell |Δ| ρ vs physics features")
        ax.legend(framealpha=0.9, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        f"Phase A4 — attribution × physics correlation  "
        f"(n_atoms = {delta_flat.size:,}, n_samples = {len(common)})",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig_path = ROOT / "paper" / "figures" / "fig_attribution_vs_physics.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
