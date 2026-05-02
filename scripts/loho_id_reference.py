"""For each LOHO host, compute the in-distribution reference MAE — the MAE
of the model trained on the full random-split data (baseline_h128_aug_long_safe)
when *evaluated on exactly the same test samples* the LOHO test uses.

This produces a fair host-level apples-to-apples degradation factor:
  degradation = LOHO MAE / in-dist MAE on the *same* host samples

(rather than dividing by the global random-split MAE, which mixes hosts)

Output: results/loho_id_reference.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402

HOSTS = ["MoS2", "Cr2I6", "C2H2", "TaSe2", "MoSSe"]
RESULTS = ROOT / "results"


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/baseline_h128_aug_long_safe.yaml"))
    cleaned = ROOT / "data/processed/cleaned_dataset.pkl"
    ds = CrystalGraphDataset(cleaned)

    model = CrystalTransformer(**cfg["model_kwargs"])
    ckpt = torch.load(ROOT / "results/baseline_h128_aug_long_safe/best.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    nmean, nstd = ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"]

    summary = {}
    for h in HOSTS:
        idxs = [i for i, s in enumerate(ds.data) if (s["metadata"].get("host") or "?") == h]
        if not idxs:
            print(f"!! no samples for host {h}")
            continue
        from torch.utils.data import Subset, DataLoader
        sub = Subset(ds, idxs)
        loader = DataLoader(sub, batch_size=64, shuffle=False, collate_fn=collate_fn)
        preds, tgts = [], []
        with torch.no_grad():
            for batch in loader:
                p = model(batch) * nstd + nmean
                preds.append(p.numpy())
                tgts.append(batch["target"].numpy())
        preds = np.concatenate(preds); tgts = np.concatenate(tgts)
        mae = float(np.abs(preds - tgts).mean())
        rmse = float(np.sqrt(((preds - tgts) ** 2).mean()))
        summary[h] = {
            "n_samples_in_host": len(idxs),
            "id_reference_mae": mae,
            "id_reference_rmse": rmse,
        }
        print(f"{h:<8} | n={len(idxs):4d}  ID-MAE = {mae:.4f}  ID-RMSE = {rmse:.4f}")

    out = RESULTS / "loho_id_reference.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
