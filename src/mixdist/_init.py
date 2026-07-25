"""Seeding helpers shared by the clusterers."""

from __future__ import annotations

from typing import Callable

import numpy as np


def kmeans_plusplus(
    n_samples: int,
    n_clusters: int,
    rng: np.random.Generator,
    dist_to_point: Callable[[int], np.ndarray],
) -> np.ndarray:
    """D^2 seeding using an arbitrary distance.

    ``dist_to_point(i)`` returns the distance from sample ``i`` to every
    sample, which lets the same routine seed Euclidean and mixed-type
    clusterers alike.  Returns the chosen sample indices.
    """
    if not 1 <= n_clusters <= n_samples:
        raise ValueError(f"n_clusters must lie in [1, {n_samples}]; got {n_clusters}.")

    chosen = np.empty(n_clusters, dtype=np.int64)
    chosen[0] = rng.integers(n_samples)
    closest = np.nan_to_num(dist_to_point(int(chosen[0])), nan=0.0)

    for c in range(1, n_clusters):
        weights = closest**2
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            # Degenerate (duplicated rows): fall back to sampling unused points.
            remaining = np.setdiff1d(np.arange(n_samples), chosen[:c])
            pool = remaining if remaining.size else np.arange(n_samples)
            chosen[c] = rng.choice(pool)
        else:
            chosen[c] = rng.choice(n_samples, p=weights / total)
        closest = np.minimum(closest, np.nan_to_num(dist_to_point(int(chosen[c])), nan=0.0))
    return chosen


def repair_empty_clusters(
    labels: np.ndarray,
    distances: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Refill empty clusters with the worst-fitting points.

    ``distances`` is ``(n, n_clusters)`` cost-to-centroid; the point with the
    largest cost to its own centroid is moved into the empty cluster.  Without
    this, ``k`` silently shrinks during iteration.
    """
    labels = labels.copy()
    own_cost = distances[np.arange(len(labels)), labels]
    for g in range(n_clusters):
        if np.any(labels == g):
            continue
        counts = np.bincount(labels, minlength=n_clusters)
        movable = np.flatnonzero(counts[labels] > 1)
        if movable.size == 0:  # pragma: no cover - fewer points than clusters
            break
        victim = movable[np.argmax(own_cost[movable])]
        labels[victim] = g
        own_cost[victim] = -np.inf
    return labels
