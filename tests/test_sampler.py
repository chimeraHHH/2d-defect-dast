from __future__ import annotations

from collections import Counter

import pytest

from src.sampler import HostBalancedSampler


class FakeDataset:
    def __init__(self):
        self.data = [
            {"metadata": {"host": "A"}},
            {"metadata": {"host": "A"}},
            {"metadata": {"host": "A"}},
            {"metadata": {"host": "B"}},
        ]


def test_host_balanced_sampler_is_epoch_addressable():
    dataset = FakeDataset()
    sampler = HostBalancedSampler(dataset, samples_per_host=4, seed=17)
    sampler.set_epoch(7)
    first = list(sampler)
    assert first == list(sampler)

    resumed = HostBalancedSampler(dataset, samples_per_host=4, seed=17)
    resumed.set_epoch(7)
    assert list(resumed) == first

    sampler.set_epoch(8)
    assert list(sampler) != first
    counts = Counter(dataset.data[index]["metadata"]["host"] for index in first)
    assert counts == {"A": 4, "B": 4}


def test_host_balanced_sampler_rejects_negative_epoch():
    sampler = HostBalancedSampler(FakeDataset(), samples_per_host=2, seed=17)
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.set_epoch(-1)
