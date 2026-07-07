"""Predict (mu, sigma_cal) on the blind cross-code OOD benchmark.

Runs the 4-seed leak-free baseline ensemble (same checkpoints + same
temperature calibration tau as scripts/predict_candidates.py, i.e. the
exact pipeline used for the paper's prospective DFT candidate selection)
on data/processed/blind_ood_benchmark.pkl (39 GPAW-computed adsorbate
defects on graphene / VS2 / CrS2 -- three hosts NOT in IMP2D).

Outputs
-------
- results/blind_ood_predictions.csv   (one row per sample, incl. DFT truth)
- results/blind_ood_predictions.json  (full record incl. per-seed preds)
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, get_atom_feature_table  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

BENCH_PATH = ROOT / "data" / "processed" / "blind_ood_benchmark.pkl"
CKPT_ROOT = ROOT / "weights&results" / "project" / "results"
RESULTS = ROOT / "results"

SEED_DIRS = [
    "baseline_h128_aug_long_safe",        # seed=42
    "baseline_h128_aug_long_safe_seed0",
    "baseline_h128_aug_long_safe_seed1",
    "baseline_h128_aug_long_safe_seed2",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model():
    return CrystalTransformer(
        atom_fea_len=9, hidden_dim=128,
        n_local_layers=3, n_global_layers=2,
        num_heads=4, dropout=0.1,
    )


def load_seed(seed_dir):
    p = CKPT_ROOT / seed_dir / "best.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    state = ck.get("model", ck.get("model_state", ck.get("state_dict", ck)))
    norm = ck.get("normalizer", {"mean": 0.0, "std": 1.0})
    model = make_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"    WARN missing={len(missing)} unexpected={len(unexpected)}")
    return model, float(norm["mean"]), float(norm["std"])


@torch.no_grad()
def predict_all(model, loader, device):
    model.to(device)
    model.eval()
    out = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        pred = model(batch)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading benchmark: {BENCH_PATH}")
    ds = CrystalGraphDataset.__new__(CrystalGraphDataset)
    with open(BENCH_PATH, "rb") as f:
        ds.data = pickle.load(f)
    ds.meta = None
    ds.atom_features = get_atom_feature_table(None)
    ds.defect_mark_neighbors = 0
    print(f"  {len(ds.data)} samples")

    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    for s in SEED_DIRS:
        print(f"  predicting with {s}")
        m, mean, std = load_seed(s)
        preds = predict_all(m, loader, device)
        preds = preds * std + mean  # de-normalise to eV
        all_preds.append(preds)
    preds = np.stack(all_preds, axis=0)            # (4, N)
    mu = preds.mean(0)
    sigma = preds.std(0)

    cal = json.load(open(RESULTS / "uq_calibration.json"))
    tau = float(cal["tau"])
    sigma_cal = sigma * tau
    print(f"  tau = {tau:.3f}")
    print(f"  mu    in [{mu.min():.3f}, {mu.max():.3f}] eV")
    print(f"  sigma_cal in [{sigma_cal.min():.3f}, {sigma_cal.max():.3f}] eV")

    rows = []
    for i, s in enumerate(ds.data):
        m = s["metadata"]
        rows.append({
            "sample_id": m["sample_id"],
            "host": m["host"],
            "dopant": m["dopant"],
            "formula": m["formula"],
            "natoms": m["natoms"],
            "traj_stage": m["traj_stage"],
            "usable_flag": m["usable_flag"],
            "E_pred_eV": round(float(mu[i]), 4),
            "sigma_raw_eV": round(float(sigma[i]), 4),
            "sigma_cal_eV": round(float(sigma_cal[i]), 4),
            "E_dft_eV": round(float(m["E_form_dft_eV"]), 4),
            "err_eV": round(float(mu[i] - m["E_form_dft_eV"]), 4),
            "per_seed": [round(float(p), 4) for p in preds[:, i]],
        })

    # CSV (without per_seed)
    csv_path = RESULTS / "blind_ood_predictions.csv"
    fields = [k for k in rows[0] if k != "per_seed"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"saved -> {csv_path}")

    # JSON (full)
    json_path = RESULTS / "blind_ood_predictions.json"
    with open(json_path, "w") as f:
        json.dump({
            "description": (
                "4-seed leak-free baseline ensemble predictions on the "
                "blind GPAW OOD benchmark (graphene/VS2/CrS2, hosts not in "
                "IMP2D). sigma_cal = tau * ensemble std, tau from "
                "uq_calibration.json."
            ),
            "tau": tau,
            "n_samples": len(rows),
            "n_seeds": len(SEED_DIRS),
            "seed_dirs": SEED_DIRS,
            "predictions": rows,
            "wall_time_min": (time.time() - t0) / 60,
        }, f, indent=2)
    print(f"saved -> {json_path}")

    # quick console table
    print(f"\n{'sid':>4} {'host':<9} {'dop':<3} {'pred':>8} {'dft':>8} "
          f"{'err':>8} {'s_cal':>7} {'flag'}")
    for r in rows:
        print(f"{r['sample_id']:>4} {r['host']:<9} {r['dopant']:<3} "
              f"{r['E_pred_eV']:>8.3f} {r['E_dft_eV']:>8.3f} "
              f"{r['err_eV']:>8.3f} {r['sigma_cal_eV']:>7.3f} "
              f"{r['usable_flag']}")

    # headline quick stats (usable only)
    ok = [r for r in rows if r["usable_flag"] == "yes"]
    err = np.array([r["err_eV"] for r in ok])
    dft = np.array([r["E_dft_eV"] for r in ok])
    pred = np.array([r["E_pred_eV"] for r in ok])
    mae = np.abs(err).mean()
    bias = err.mean()
    pear = np.corrcoef(pred, dft)[0, 1]
    print(f"\n[usable n={len(ok)}]  raw MAE = {mae:.3f} eV | "
          f"mean bias = {bias:+.3f} eV | Pearson = {pear:+.3f}")
    for host in ("graphene", "VS2", "CrS2"):
        sel = [r for r in ok if r["host"] == host]
        e = np.array([r["err_eV"] for r in sel])
        print(f"  {host:<9} n={len(sel):>2}  MAE={np.abs(e).mean():.3f}  "
              f"bias={e.mean():+.3f}")


if __name__ == "__main__":
    main()
