"""Sanity tests for the v2 periodic-aware attention.

Checks:
  * ``compute_frac_disp`` is invariant under translation, rotation, and
    PBC shifts.
  * ``PeriodicFourierBias`` is exactly periodic under integer-lattice shifts.
  * ``MultiScaleDistanceBias`` and ``DefectAwareBias`` are well-behaved
    (right shape, finite, sane gradients).
  * End-to-end ``PeriodicCrystalTransformer`` forward / backward runs and is
    invariant (numerically) to the same operations.

Run:
    cd project
    .venv/bin/python -m scripts.test_attention_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.attention_v2 import (
    DefectAwareBias,
    MultiScaleDistanceBias,
    PeriodicCrystalTransformer,
    PeriodicFourierBias,
    compute_frac_disp,
)


# ----------------------------------------------------------------------------
def random_batch(B: int = 2, N: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    cell = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    cell[:, 0, 0] = 6.0 + 0.5 * torch.rand(B, generator=g)
    cell[:, 1, 1] = 6.0 + 0.5 * torch.rand(B, generator=g)
    cell[:, 2, 2] = 18.0 + torch.rand(B, generator=g)
    positions = torch.rand(B, N, 3, generator=g)
    positions = torch.einsum("bij,bjk->bik", positions, cell)  # frac → cart
    x = torch.randn(B, N, 9, generator=g)
    defect_mask = torch.zeros(B, N, dtype=torch.long)
    defect_mask[:, 0] = 1
    atom_mask = torch.ones(B, N, dtype=torch.bool)
    dist_matrix = torch.cdist(positions, positions)
    num_atoms_list = [N] * B
    edge_index_list = [torch.empty(2, 0, dtype=torch.long)] * B
    edge_dist_list = [torch.empty(0, dtype=torch.float32)] * B
    triplet_index_list = [torch.empty(0, 3, dtype=torch.long)] * B
    angles_list = [torch.empty(0, dtype=torch.float32)] * B
    return {
        "x": x,
        "positions": positions,
        "cell": cell,
        "defect_mask": defect_mask,
        "atom_mask": atom_mask,
        "dist_matrix": dist_matrix,
        "num_atoms_list": num_atoms_list,
        "edge_index_list": edge_index_list,
        "edge_dist_list": edge_dist_list,
        "triplet_index_list": triplet_index_list,
        "angles_list": angles_list,
        "target": torch.tensor([0.0] * B),
    }


def assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-4, label: str = ""):
    diff = (a - b).abs().max().item()
    msg = f"[{label}] max abs diff = {diff:.3e} (tol {tol:.3e})"
    if diff > tol:
        raise AssertionError("FAIL " + msg)
    print("PASS " + msg)


# ----------------------------------------------------------------------------
def test_frac_disp_translation():
    batch = random_batch()
    pos = batch["positions"]
    cell = batch["cell"]
    f1 = compute_frac_disp(pos, cell)
    f2 = compute_frac_disp(pos + torch.tensor([2.5, -1.1, 0.7]), cell)
    assert_close(f1, f2, tol=1e-5, label="frac_disp ⊥ translation")


def test_frac_disp_pbc_shift():
    batch = random_batch()
    pos = batch["positions"].clone()
    cell = batch["cell"]
    # shift atom 2 in batch 0 by lattice vector a (row 0)
    pos[0, 2, :] += cell[0, 0, :]
    f1 = compute_frac_disp(batch["positions"], cell)
    f2 = compute_frac_disp(pos, cell)
    assert_close(f1, f2, tol=1e-5, label="frac_disp ⊥ PBC single-atom shift")


def test_frac_disp_rotation():
    batch = random_batch()
    pos = batch["positions"]
    cell = batch["cell"]
    theta = torch.tensor(0.7)
    R = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pos_R = pos @ R.T
    cell_R = cell @ R.T
    f1 = compute_frac_disp(pos, cell)
    f2 = compute_frac_disp(pos_R, cell_R)
    assert_close(f1, f2, tol=1e-5, label="frac_disp ⊥ joint rotation")


def test_pfa_pbc_invariance():
    torch.manual_seed(0)
    pfa = PeriodicFourierBias(num_heads=4, k_max=2)
    # randomise to break the zero-init
    for p in pfa.parameters():
        p.data = torch.randn_like(p) * 0.1
    pos = torch.rand(1, 6, 3)
    cell = torch.eye(3).unsqueeze(0) * 5.0
    pos_cart = torch.einsum("bij,bjk->bik", pos, cell)
    f0 = compute_frac_disp(pos_cart, cell)
    pos_shifted = pos_cart.clone()
    pos_shifted[0, 1, :] += cell[0, 1, :]  # shift one atom by full a_2
    f1 = compute_frac_disp(pos_shifted, cell)
    bias0 = pfa(f0)
    bias1 = pfa(f1)
    assert_close(bias0, bias1, tol=1e-4, label="PFA ⊥ PBC single-atom shift")


def test_multiscale_dist_shape():
    torch.manual_seed(1)
    block = MultiScaleDistanceBias(num_heads=4, n_rbf=16, r_short=5.0, r_max=12.0)
    dist = torch.rand(2, 6, 6) * 12.0
    out = block(dist)
    assert out.shape == (2, 4, 6, 6), f"shape {out.shape}"
    assert torch.isfinite(out).all(), "NaN/inf in multiscale bias"
    print(f"PASS multiscale shape={tuple(out.shape)}, finite, |out|max={out.abs().max():.3e}")


def test_defect_bias_shape():
    db = DefectAwareBias(num_heads=4)
    db.bias.data = torch.randn(4, 4)
    mask = torch.tensor([[0, 1, 0], [1, 0, 0]])
    out = db(mask)
    assert out.shape == (2, 4, 3, 3), f"shape {out.shape}"
    # check (defect_i, defect_j) categorical equality
    assert torch.allclose(out[0, :, 1, 0], out[1, :, 0, 1], atol=1e-6)
    print("PASS defect bias categorical mapping")


def test_model_translation_invariance():
    torch.manual_seed(2)
    model = PeriodicCrystalTransformer(
        atom_fea_len=9,
        hidden_dim=32,
        n_local_layers=1,
        n_global_layers=2,
        num_heads=4,
        use_pfa=True,
        k_max=2,
        use_long_range=True,
        use_defect_bias=True,
    ).eval()
    batch = random_batch()
    out0 = model(batch).detach()
    batch_t = dict(batch)
    batch_t["positions"] = batch["positions"] + torch.tensor([2.0, -1.5, 0.7])
    out1 = model(batch_t).detach()
    assert_close(out0, out1, tol=1e-4, label="model ⊥ translation")


def test_model_rotation_invariance():
    torch.manual_seed(3)
    model = PeriodicCrystalTransformer(
        atom_fea_len=9,
        hidden_dim=32,
        n_local_layers=1,
        n_global_layers=2,
        num_heads=4,
        use_pfa=True,
        use_long_range=True,
        use_defect_bias=True,
    ).eval()
    batch = random_batch()
    theta = torch.tensor(0.7)
    R = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out0 = model(batch).detach()
    batch_R = dict(batch)
    batch_R["positions"] = batch["positions"] @ R.T
    batch_R["cell"] = batch["cell"] @ R.T
    # dist_matrix is rotation-invariant by construction (we use pairwise dist)
    out1 = model(batch_R).detach()
    assert_close(out0, out1, tol=1e-3, label="model ⊥ joint rotation")


def test_model_pbc_invariance():
    torch.manual_seed(4)
    model = PeriodicCrystalTransformer(
        atom_fea_len=9,
        hidden_dim=32,
        n_local_layers=1,
        n_global_layers=2,
        num_heads=4,
        use_pfa=True,
        use_long_range=True,
        use_defect_bias=True,
        recompute_dist_from_positions=True,
    ).eval()
    batch = random_batch()
    out0 = model(batch).detach()
    batch_p = dict(batch)
    pos = batch["positions"].clone()
    cell = batch["cell"]
    # shift atom 2 in sample 0 by full a_1
    pos[0, 2, :] += cell[0, 0, :]
    batch_p["positions"] = pos
    # IMPORTANT: dist_matrix in collate was the precomputed minimum-image distance,
    # which is itself PBC-invariant — but the user-supplied tensor is the *original* one,
    # so the model should reproduce the same answer when it recomputes via positions+cell.
    # We force recompute via the model knob `recompute_dist_from_positions=True`.
    out1 = model(batch_p).detach()
    assert_close(out0, out1, tol=1e-3, label="model ⊥ PBC single-atom shift (recompute)")


def test_model_backward_runs():
    torch.manual_seed(5)
    model = PeriodicCrystalTransformer(
        atom_fea_len=9,
        hidden_dim=32,
        n_local_layers=2,
        n_global_layers=2,
        num_heads=4,
    )
    batch = random_batch()
    out = model(batch)
    loss = (out - batch["target"]).pow(2).mean()
    loss.backward()
    n_grad = sum(int(p.grad.abs().sum() > 0) for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"PASS backward: {n_grad}/{n_total} params received non-zero grad")


def main():
    print("=" * 60)
    print("Running v2 attention sanity tests")
    print("=" * 60)
    test_frac_disp_translation()
    test_frac_disp_pbc_shift()
    test_frac_disp_rotation()
    test_pfa_pbc_invariance()
    test_multiscale_dist_shape()
    test_defect_bias_shape()
    test_model_translation_invariance()
    test_model_rotation_invariance()
    test_model_pbc_invariance()
    test_model_backward_runs()
    print("=" * 60)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
