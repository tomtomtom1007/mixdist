"""Per-column dispersion statistics.

These answer one question: *how much distance does this column contribute on
average?*  That number drives :mod:`mixdist.weighting`, and it is the whole
reason a 50-level ``customer_id``-ish column can drown out ``age`` under plain
Gower — see :func:`expected_dissimilarity`.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np


def gini_mean_difference(x: np.ndarray) -> float:
    r"""Exact :math:`E|X - X'|` for two i.i.d. draws from the empirical law of ``x``.

    Computed in :math:`O(n \log n)` from the order statistics,

    .. math:: \mathrm{GMD} = \frac{2}{n^2} \sum_{i=1}^{n} (2i - n - 1) x_{(i)},

    rather than by materialising the :math:`n^2` pairwise differences.
    ``nan`` values are dropped.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    n = values.size
    if n < 2:
        return 0.0
    values.sort()
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(2.0 * np.dot(2.0 * i - n - 1.0, values) / (n * n))


def gini_impurity(codes: np.ndarray, *, n_levels: int | None = None) -> float:
    r"""Exact :math:`P(X \neq X')` for the empirical law of a categorical column.

    Equal to :math:`1 - \sum_v p_v^2`; ``-1`` codes are treated as missing.
    """
    values = np.asarray(codes).ravel()
    values = values[values >= 0]
    if values.size < 2:
        return 0.0
    minlength = 0 if n_levels is None else int(n_levels)
    counts = np.bincount(values.astype(np.int64), minlength=minlength).astype(np.float64)
    p = counts / counts.sum()
    return float(1.0 - np.dot(p, p))


def expected_dissimilarity(num: np.ndarray, cat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Mean per-column dissimilarity between two random rows.

    Returns ``(d_num, d_cat)``, aligned with the encoded column slots.

    This makes the "categorical swamps numeric" failure mode quantitative.  For
    a range-scaled numeric column the expected contribution is at most
    :math:`1/3` (uniform) and typically nearer :math:`0.2`; for a balanced
    nominal column with :math:`K` levels it is :math:`1 - 1/K`, i.e. ``0.5`` at
    two levels and ``0.98`` at fifty.  Under equal weights a high-cardinality
    nominal column therefore contributes several times the distance of a
    numeric one *regardless of whether it carries any signal*.
    """
    num = np.asarray(num, dtype=np.float64)
    cat = np.asarray(cat)
    d_num = np.array([gini_mean_difference(num[:, j]) for j in range(num.shape[1])])
    d_cat = np.array([gini_impurity(cat[:, j]) for j in range(cat.shape[1])])
    return d_num, d_cat


@contextmanager
def _quiet():
    """Silence the all-NaN-slice warnings; callers handle the ``nan`` explicitly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            yield


def nanmean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``np.nanmean`` without the all-NaN-slice warning."""
    with _quiet():
        return np.nanmean(x, axis=axis)


def nanstd(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``np.nanstd`` without the all-NaN-slice warning."""
    with _quiet():
        return np.nanstd(x, axis=axis)


def nanmedian(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``np.nanmedian`` without the all-NaN-slice warning."""
    with _quiet():
        return np.nanmedian(x, axis=axis)
