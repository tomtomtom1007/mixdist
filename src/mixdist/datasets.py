"""Synthetic mixed-type data, including the failure case this library targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_mixed_blobs"]


def make_mixed_blobs(
    n_samples: int = 600,
    *,
    n_clusters: int = 3,
    n_numeric: int = 3,
    n_nominal: int = 2,
    n_noise_nominal: int = 2,
    nominal_levels: int = 3,
    noise_cardinality: int = 30,
    separation: float = 2.5,
    purity: float = 0.85,
    missing_rate: float = 0.0,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a mixed-type table with known cluster structure.

    The defaults deliberately reproduce the situation that motivates
    ``weights="balanced"``: alongside the informative columns there are
    ``n_noise_nominal`` **high-cardinality categorical columns carrying no
    signal at all**.  Under equal weights each of those contributes an expected
    dissimilarity of about ``1 - 1 / noise_cardinality`` (≈ 0.97 at the default
    cardinality) against roughly 0.2 for an informative numeric column, so the
    noise dominates the distance and the true clusters dissolve.

    Parameters
    ----------
    n_samples, n_clusters : int
    n_numeric : int
        Informative Gaussian columns, named ``num_0`` ...
    n_nominal : int
        Informative categorical columns, named ``cat_0`` ...
    n_noise_nominal : int
        Uninformative high-cardinality columns, named ``noise_0`` ...
    nominal_levels : int
        Levels per informative categorical column.
    noise_cardinality : int
        Levels per noise column.
    separation : float
        Distance between adjacent cluster means in numeric space (unit variance).
    purity : float
        Probability that an informative categorical takes its cluster's
        preferred level; the remainder is spread over the other levels.
    missing_rate : float
        Fraction of cells blanked out, to exercise missing-value handling.
    random_state : int, optional

    Returns
    -------
    X : pandas.DataFrame
    y : ndarray of shape (n_samples,)
        Ground-truth cluster assignment.

    Examples
    --------
    >>> X, y = make_mixed_blobs(n_samples=120, random_state=0)
    >>> sorted(X.columns)[:2]
    ['cat_0', 'cat_1']
    """
    if not 0.0 <= purity <= 1.0:
        raise ValueError("purity must lie in [0, 1].")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must lie in [0, 1).")
    if nominal_levels < 2 or noise_cardinality < 2:
        raise ValueError("nominal_levels and noise_cardinality must be >= 2.")

    rng = np.random.default_rng(random_state)
    y = rng.integers(0, n_clusters, size=n_samples)
    data: dict[str, np.ndarray] = {}

    centres = rng.normal(size=(n_clusters, n_numeric)) if n_numeric else np.zeros((n_clusters, 0))
    if n_numeric:
        centres = separation * centres / np.maximum(
            np.linalg.norm(centres, axis=1, keepdims=True), 1e-12
        )
        values = centres[y] + rng.normal(size=(n_samples, n_numeric))
        for j in range(n_numeric):
            data[f"num_{j}"] = values[:, j]

    for j in range(n_nominal):
        preferred = (np.arange(n_clusters) + j) % nominal_levels
        off = (1.0 - purity) / max(nominal_levels - 1, 1)
        probs = np.full((n_clusters, nominal_levels), off)
        probs[np.arange(n_clusters), preferred] = purity
        draws = np.array([rng.choice(nominal_levels, p=probs[g]) for g in y])
        data[f"cat_{j}"] = np.array([f"L{v}" for v in draws], dtype=object)

    for j in range(n_noise_nominal):
        draws = rng.integers(0, noise_cardinality, size=n_samples)
        data[f"noise_{j}"] = np.array([f"N{v}" for v in draws], dtype=object)

    X = pd.DataFrame(data)
    if missing_rate > 0:
        mask = rng.random(X.shape) < missing_rate
        X = X.mask(pd.DataFrame(mask, columns=X.columns, index=X.index))
    return X, y
