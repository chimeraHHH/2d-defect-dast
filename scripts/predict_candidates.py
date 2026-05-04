"""C17 Stage 2: Predict (μ, σ) for candidates using 4-seed ensemble.

Loads the 4 leak-free baseline checkpoints and runs each on the
generated candidate set, producing per-sample mean and std.

Outputs
-------
- results/candidates_c17_predictions.json  (μ, σ per candidate)
"""
from __future__ import annotations

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

from src.dataset import CrystalGraphDataset, collate_fn  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

CAND_PATH = ROOT / "data" / "processed" / "candidates_c17.pkl"
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
    p = ROOT / "results" / seed_dir / "best.pt"
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

    print(f"Loading candidates: {CAND_PATH}")
    cls = CrystalGraphDataset.__new__(CrystalGraphDataset)
    with open(CAND_PATH, "rb") as f:
        cls.data = pickle.load(f)
    cls.meta = None
    from src.dataset import get_atom_feature_table
    cls.atom_features = get_atom_feature_table(None)
    cls.defect_mark_neighbors = 0
    print(f"  {len(cls.data)} candidates")

    loader = DataLoader(cls, batch_size=8, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    for s in SEED_DIRS:
        print(f"  predicting with {s}")
        m, mean, std = load_seed(s)
        preds = predict_all(m, loader, device)
        # de-normalise back to eV
        preds = preds * std + mean
        all_preds.append(preds)
    preds = np.stack(all_preds, axis=0)            # (4, N)
    mu = preds.mean(0)
    sigma = preds.std(0)
    print(f"  μ ∈ [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  σ ∈ [{sigma.min():.4f}, {sigma.max():.4f}]")

    # apply temperature scaling from §5.9 (τ from uq_calibration.json)
    cal = json.load(open(RESULTS / "uq_calibration.json"))
    tau = float(cal["tau"])
    sigma_cal = sigma * tau
    print(f"  τ = {tau:.3f}")

    out = []
    for i, s in enumerate(cls.data):
        out.append({
            "id": int(i),
            "host": s["metadata"]["host"],
            "dopant": s["metadata"]["dopant"],
            "defect_type": s["metadata"]["defecttype"],
            "natoms": int(s["metadata"]["natoms"]),
            "mu": float(mu[i]),
            "sigma_raw": float(sigma[i]),
            "sigma_cal": float(sigma_cal[i]),
        })

    out_path = RESULTS / "candidates_c17_predictions.json"
    with open(out_path, "w") as f:
        json.dump({
            "tau": tau,
            "n_candidates": len(out),
            "n_seeds": len(SEED_DIRS),
            "predictions": out,
            "wall_time_min": (time.time() - t0) / 60,
        }, f, indent=2)
    print(f"saved -> {out_path}")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
