"""Dataset class and collate function for the defect formation-energy task."""
from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.features import get_atom_feature_table


class CrystalGraphDataset(Dataset):
    """Loads the cleaned defect dataset and exposes per-sample tensors.

    Each `__getitem__` returns a dict that already lives in CPU memory; the
    DataLoader's worker can move tensors to GPU via the collate fn.
    """

    def __init__(
        self,
        data_path: str | Path,
        feature_table_path: Optional[str | Path] = None,
        defect_mark_neighbors: int = 0,
    ) -> None:
        super().__init__()
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            self.data: List[Dict[str, Any]] = pickle.load(f)
        # default to the reference min-max normalised feature table (better
        # convergence than the home-grown z-score variant in src/features.py).
        if feature_table_path is None:
            ref = Path(__file__).resolve().parent.parent / "data" / "atom_features_ref.pth"
            if ref.exists():
                feature_table_path = ref
        self.atom_features = get_atom_feature_table(feature_table_path)
        self.defect_mark_neighbors = defect_mark_neighbors

        # build defect-mark cache once: which atom index is the dopant?
        for sample in self.data:
            sample["defect_mask"] = self._compute_defect_mask(sample)

    # ------------------------------------------------------------------ helpers
    def _compute_defect_mask(self, sample: Dict[str, Any]) -> np.ndarray:
        """Heuristic: mark the defect atom in IMP2D supercells.

        IMP2D defects are constructed by ASE's ``DefectBuilder`` which appends
        the dopant atom at the end of the positions list. We mark the LAST
        atom whose element matches the dopant tag in metadata; if the dopant
        is unique to the host (e.g. SnS2:Cl) this picks the only candidate.
        For self-substitution / anti-site defects (e.g. MoTe2:Te) the heuristic
        still localises to one atom, biased toward the inserted one.
        """
        natoms = len(sample["numbers"])
        mask = np.zeros(natoms, dtype=np.int64)
        dopant = sample["metadata"].get("dopant", "")
        if not dopant:
            return mask
        try:
            from ase.data import atomic_numbers as _AZ

            z = _AZ.get(dopant, None)
        except Exception:  # pragma: no cover - defensive
            z = None
        if z is None:
            return mask
        candidates = np.flatnonzero(sample["numbers"] == z)
        if candidates.size == 0:
            return mask
        # mark the last candidate (DefectBuilder convention)
        mask[candidates[-1]] = 1
        return mask

    # ------------------------------------------------------------------ pytorch
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        numbers = torch.from_numpy(sample["numbers"])
        x = self.atom_features[numbers]
        defect_mask = torch.from_numpy(sample["defect_mask"]).long()
        return {
            "x": x,
            "defect_mask": defect_mask,
            "edge_index": torch.from_numpy(sample["edge_index"]),
            "edge_dist": torch.from_numpy(sample["edge_dist"]),
            "triplet_index": torch.from_numpy(sample["triplet_index"]),
            "angles": torch.from_numpy(sample["angles"]),
            "dist_matrix": torch.from_numpy(sample["dist_matrix"]),
            "positions": torch.from_numpy(sample["positions"]),
            "cell": torch.from_numpy(sample["cell"]),
            "target": torch.tensor(sample["target"], dtype=torch.float32),
            "num_atoms": numbers.numel(),
        }


# --------------------------------------------------------------------- collate
def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pack variable-size graphs into padded batched tensors."""
    batch_size = len(batch)
    natoms_list = [item["num_atoms"] for item in batch]
    n_max = max(natoms_list)
    feat_dim = batch[0]["x"].shape[-1]

    x = torch.zeros(batch_size, n_max, feat_dim, dtype=torch.float32)
    defect_mask = torch.zeros(batch_size, n_max, dtype=torch.long)
    atom_mask = torch.zeros(batch_size, n_max, dtype=torch.bool)
    dist_matrix = torch.zeros(batch_size, n_max, n_max, dtype=torch.float32)
    target = torch.zeros(batch_size, dtype=torch.float32)
    num_atoms = torch.zeros(batch_size, dtype=torch.long)

    edge_index_list, edge_dist_list = [], []
    triplet_index_list, angles_list = [], []
    cell = torch.zeros(batch_size, 3, 3, dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item["num_atoms"]
        x[i, :n] = item["x"]
        defect_mask[i, :n] = item["defect_mask"]
        atom_mask[i, :n] = True
        dist_matrix[i, :n, :n] = item["dist_matrix"]
        target[i] = item["target"]
        num_atoms[i] = n
        cell[i] = item["cell"]
        edge_index_list.append(item["edge_index"])
        edge_dist_list.append(item["edge_dist"])
        triplet_index_list.append(item["triplet_index"])
        angles_list.append(item["angles"])

    return {
        "x": x,
        "defect_mask": defect_mask,
        "atom_mask": atom_mask,
        "dist_matrix": dist_matrix,
        "cell": cell,
        "target": target,
        "num_atoms": num_atoms,
        "num_atoms_list": natoms_list,
        "edge_index_list": edge_index_list,
        "edge_dist_list": edge_dist_list,
        "triplet_index_list": triplet_index_list,
        "angles_list": angles_list,
    }


# --------------------------------------------------------------------- split
def split_indices(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )


def make_splits(
    dataset: CrystalGraphDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    train_idx, val_idx, test_idx = split_indices(
        len(dataset), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def host_aware_splits(
    dataset: CrystalGraphDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Split by host material so the test set contains unseen host crystals."""
    rng = random.Random(seed)
    by_host = defaultdict(list)
    for idx, sample in enumerate(dataset.data):
        host = sample["metadata"].get("host", "unknown") or "unknown"
        by_host[host].append(idx)
    hosts = sorted(by_host.keys())
    rng.shuffle(hosts)
    n_total = sum(len(v) for v in by_host.values())
    train_target = int(train_ratio * n_total)
    val_target = int(val_ratio * n_total)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for host in hosts:
        idxs = by_host[host]
        if len(train_idx) + len(idxs) <= train_target:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= val_target:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    if not test_idx:
        # fallback: take last 10% of train into test
        n = max(1, len(train_idx) // 10)
        test_idx = train_idx[-n:]
        train_idx = train_idx[:-n]
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
