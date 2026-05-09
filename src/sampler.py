"""Host-balanced sampler for defect datasets.

In IMP2D, host material counts range from 1 (Ti2CO2) to 464 (MoSSe).
A naive random sampler over-represents large hosts. This sampler ensures
each host contributes roughly equally per epoch, improving coverage of
the chemical space without requiring extra data.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler, Subset


class HostBalancedSampler(Sampler[int]):
    """Yields indices so every host is sampled equally per epoch.

    For each epoch: pick ``samples_per_host`` indices from each host
    (with replacement if the host has fewer). The epoch size is
    ``n_hosts × samples_per_host``.
    """

    def __init__(
        self,
        dataset,
        subset_indices: Optional[List[int]] = None,
        samples_per_host: int = 50,
        seed: int = 42,
    ) -> None:
        if isinstance(dataset, Subset):
            base_data = dataset.dataset.data
            indices = dataset.indices
        else:
            base_data = dataset.data
            indices = list(range(len(base_data)))

        if subset_indices is not None:
            indices = subset_indices

        self.host_to_indices: dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            sample = base_data[idx]
            host = sample.get("metadata", {}).get("host", "unknown") or "unknown"
            self.host_to_indices[host].append(idx)

        self.hosts = sorted(self.host_to_indices.keys())
        self.samples_per_host = samples_per_host
        self.rng = random.Random(seed)
        self._epoch_size = len(self.hosts) * samples_per_host

    def __iter__(self) -> Iterator[int]:
        indices = []
        for host in self.hosts:
            pool = self.host_to_indices[host]
            if len(pool) >= self.samples_per_host:
                chosen = self.rng.sample(pool, self.samples_per_host)
            else:
                chosen = self.rng.choices(pool, k=self.samples_per_host)
            indices.extend(chosen)
        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self._epoch_size
