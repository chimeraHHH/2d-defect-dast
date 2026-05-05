"""Phase A2 — per-atom occlusion attribution arrays for the test fold.

Simple per-sample loop: for each test sample, mask each atom one at a
time and re-evaluate. Saves the per-atom Δ array for every sample to a
single NPZ.

Set MAX_SAMPLES env var to limit how many test samples to process
(default: full test fold = 1065).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, collate_fn, make_splits  # noqa: E402
from src.models import CrystalTransformer  # noqa: E402


def _to(d, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def occlusion_per_atom_simple(model, sample_dict, device, mean_, std_):
    """For each atom k in this sample, mask atom k by setting
    atom_mask[batch=0, k]=False and re-evaluate. Returns (pred_full eV, |Δ_i| eV array).
    """
    n = sample_dict["num_atoms"]
    base_batch = collate_fn([sample_dict])
    base_batch = _to(base_batch, device)
    with torch.no_grad():
        pred_full_n = model(base_batch).cpu().numpy().item()
    pred_full = pred_full_n * std_ + mean_

    deltas = np.zeros(n, dtype=np.float32)
    for k in range(n):
        am = base_batch["atom_mask"].clone()
        am[0, k] = False
        modified = dict(base_batch)
        modified["atom_mask"] = am
        with torch.no_grad():
            pred_n = model(modified).cpu().numpy().item()
        pred_eV = pred_n * std_ + mean_
        deltas[k] = abs(pred_full - pred_eV)
    return pred_full, deltas


def main():
    cfg_path = ROOT / "configs" / "baseline_h128_aug_long_safe.yaml"
    ckpt_path = ROOT / "results" / "baseline_h128_aug_long_safe" / "best.pt"
    cfg = yaml.safe_load(open(cfg_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    nmean = ckpt["normalizer"]["mean"]
    nstd = ckpt["normalizer"]["std"]
    print(f"normalizer: mean={nmean:.4f}  std={nstd:.4f}", flush=True)

    ds_path = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
    print(f"loading {ds_path}", flush=True)
    ds = CrystalGraphDataset(ds_path)
    _, _, test_set = make_splits(ds, 0.8, 0.1, 42)
    print(f"loaded test fold: {len(test_set)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    model = CrystalTransformer(**cfg["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"model loaded", flush=True)

    max_n = int(os.environ.get("MAX_SAMPLES", "0")) or len(test_set)
    print(f"will process up to {max_n} samples", flush=True)

    pred_fulls = []
    targets = []
    sample_ids = []
    flat_delta = []
    sample_offsets = [0]

    t0 = time.time()
    n_done = 0
    for i in range(min(max_n, len(test_set))):
        if i < 3 or i % 5 == 0:
            print(f"  [enter sample {i}]  elapsed={time.time()-t0:.1f}s", flush=True)
        s = test_set[i]
        if s["defect_mask"].sum().item() < 1:
            continue
        if s["num_atoms"] > 100:
            continue
        if i < 3:
            print(f"    n_atoms={s['num_atoms']}, calling occlusion...", flush=True)
        try:
            pred_full, delta = occlusion_per_atom_simple(model, s, device, nmean, nstd)
        except Exception as e:
            print(f"  sample {i} FAILED: {e}", flush=True)
            continue
        target = float(s["target"].item())
        pred_fulls.append(pred_full)
        targets.append(target)
        sample_ids.append(test_set.indices[i])
        flat_delta.append(delta)
        sample_offsets.append(sample_offsets[-1] + delta.shape[0])
        n_done += 1
        if n_done <= 5 or n_done % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-3)
            eta = (max_n - n_done) / max(rate, 1e-3)
            print(f"  {n_done}/{max_n} done  ({elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s)", flush=True)

    flat_delta = np.concatenate(flat_delta) if flat_delta else np.array([])
    out = ROOT / "results" / "phase_a_occlusion_per_atom.npz"
    np.savez(
        out,
        sample_offsets=np.array(sample_offsets, dtype=np.int64),
        delta_per_atom=flat_delta,
        pred_full=np.array(pred_fulls, dtype=np.float32),
        target=np.array(targets, dtype=np.float32),
        sample_id=np.array(sample_ids, dtype=np.int64),
    )
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"  N samples = {n_done}", flush=True)
    print(f"  N atoms   = {flat_delta.shape[0]}", flush=True)
    print(f"  total wall = {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
