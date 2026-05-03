"""C12-followup: retry ViSNet with safer hyperparameters (lr=1e-4, clip 1.0).

The default lr=5e-4 destabilises ViSNet (NaN by ep 10). This re-runs
with the recommended ViSNet defaults from the original paper.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CrystalGraphDataset, split_indices  # noqa: E402

# load patched radius_graph (inline so we don't need package import)
def _radius_graph_pure(x, r, batch=None, loop=False, max_num_neighbors=32,
                       flow="source_to_target", num_workers=None):
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    N = x.size(0)
    d2 = torch.cdist(x, x)
    same = batch.unsqueeze(0) == batch.unsqueeze(1)
    if not loop:
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        same = same & ~eye
    valid = same & (d2 <= r)
    if max_num_neighbors > 0:
        d_for_sort = torch.where(valid, d2, torch.full_like(d2, float("inf")))
        _, topk = torch.topk(d_for_sort, k=min(max_num_neighbors, N),
                              dim=1, largest=False)
        keep = torch.zeros_like(valid)
        keep.scatter_(1, topk, True)
        valid = valid & keep
    rows, cols = valid.nonzero(as_tuple=True)
    return torch.stack([rows, cols], dim=0)
import torch_geometric.nn.pool as _pp; _pp.radius_graph = _radius_graph_pure  # noqa: E402
import torch_geometric.nn.models.visnet as _vm; _vm.radius_graph = _radius_graph_pure  # noqa: E402


def collate_pyg(samples):
    z_all, pos_all, batch_idx, targets = [], [], [], []
    for i, s in enumerate(samples):
        z = torch.as_tensor(s["numbers"], dtype=torch.long)
        pos = torch.as_tensor(s["positions"], dtype=torch.float32)
        z_all.append(z); pos_all.append(pos)
        batch_idx.append(torch.full((z.numel(),), i, dtype=torch.long))
        targets.append(float(s["target"]))
    return (torch.cat(z_all), torch.cat(pos_all),
            torch.cat(batch_idx), torch.tensor(targets, dtype=torch.float32))

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.pkl"
RESULTS = ROOT / "results"

EPOCHS = 80
BATCH_SIZE = 16     # smaller for stability
LR = 1e-4           # lower LR for equivariant model
WEIGHT_DECAY = 1e-4
CLIP = 1.0          # tighter clip
PATIENCE = 12
SEED = 42


def evaluate(model, loader, device):
    model.eval()
    err, n = 0.0, 0
    with torch.no_grad():
        for z, pos, b, y in loader:
            z, pos, b, y = z.to(device), pos.to(device), b.to(device), y.to(device)
            x, _v = model(z, pos, b)
            pred = x.squeeze(-1)
            err += (pred - y).abs().sum().item()
            n += y.numel()
    return err / max(n, 1)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(SEED)

    dataset = CrystalGraphDataset(DATA_PATH)
    train_idx, val_idx, test_idx = split_indices(len(dataset), 0.8, 0.1, SEED)
    train_set = [dataset.data[i] for i in train_idx]
    val_set = [dataset.data[i] for i in val_idx]
    test_set = [dataset.data[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_pyg)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_pyg)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_pyg)

    from torch_geometric.nn.models import ViSNet
    model = ViSNet(
        lmax=1, num_heads=4, num_layers=4,        # smaller for stability
        hidden_channels=128, num_rbf=32,
        cutoff=5.0, max_num_neighbors=32,
        std=1.0, mean=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViSNet (safer config): {n_params/1e6:.3f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.SmoothL1Loss()

    best_val, bad = float("inf"), 0
    best_state = None
    for ep in range(EPOCHS):
        ep_start = time.time()
        model.train()
        n_skip = 0
        for z, pos, b, y in train_loader:
            z, pos, b, y = z.to(device), pos.to(device), b.to(device), y.to(device)
            x, _v = model(z, pos, b)
            pred = x.squeeze(-1)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                n_skip += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        sched.step()
        val_mae = evaluate(model, val_loader, device)
        if not (val_mae == val_mae):
            print(f"  ep {ep}: NaN val — abort"); break
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep < 3 or ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:2d}  val_mae={val_mae:.4f}  skip={n_skip}  "
                  f"({time.time()-ep_start:.0f}s)", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop at ep {ep}"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader, device)
    print(f"\nViSNet retry — Test MAE = {test_mae:.4f}  ({(time.time()-t0)/60:.1f} min)")

    # patch into existing JSON
    p = RESULTS / "gnn_baselines.json"
    with open(p) as f:
        data = json.load(f)
    for r in data["results"]:
        if r.get("model") == "ViSNet":
            r["test_mae"] = float(test_mae)
            r["best_val_mae"] = float(best_val)
            r["wall_sec"] = float(time.time() - t0)
            r["epochs_run"] = ep + 1
            r["note"] = "lr=1e-4, clip=1.0, bs=16, lmax=1, 4 layers"
            break
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"updated {p}")


if __name__ == "__main__":
    main()
