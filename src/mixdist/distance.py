"""Chunked Gower distance kernels.

The whole file exists to avoid one thing: materialising an ``(n, m, p)``
intermediate.  Distances are accumulated **one column at a time** into an
``(n_block, m_block)`` buffer, so peak memory is set by the block size rather
than by the number of variables.  That is what makes 10^5-row tables tractable.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = ["block_sums", "pairwise", "iter_chunks", "topk", "column_contributions"]

#: Target elements per distance block (~64 MB of float64).
DEFAULT_BLOCK = 8_000_000


def _rows_per_block(n_cols: int, block_elements: int) -> int:
    return max(1, int(block_elements // max(n_cols, 1)))


def block_sums(
    a_num: np.ndarray,
    a_cat: np.ndarray,
    b_num: np.ndarray,
    b_cat: np.ndarray,
    w_num: np.ndarray,
    w_cat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted dissimilarity sum and weight sum for every pair in the block.

    Returns ``(S, W)`` with ``S[i, j] = sum_k w_k * delta_ijk * d_k(i, j)`` and
    ``W[i, j] = sum_k w_k * delta_ijk``, where ``delta_ijk`` is 0 when column
    ``k`` is missing in either row.  The Gower distance is ``S / W``.
    """
    na, nb = a_num.shape[0], b_num.shape[0]
    S = np.zeros((na, nb), dtype=np.float64)
    W = np.zeros((na, nb), dtype=np.float64)

    for k, w in enumerate(w_num):
        if w == 0.0:
            continue
        # nan propagates through the subtraction, which is exactly the missing mask.
        diff = np.abs(a_num[:, k][:, None] - b_num[:, k][None, :])
        ok = ~np.isnan(diff)
        S += w * np.where(ok, diff, 0.0)
        W += w * ok

    for k, w in enumerate(w_cat):
        if w == 0.0:
            continue
        ak = a_cat[:, k][:, None]
        bk = b_cat[:, k][None, :]
        ok = (ak >= 0) & (bk >= 0)
        neq = (ak != bk) & ok
        S += w * neq
        W += w * ok

    return S, W


def _normalise(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    out = np.full(S.shape, np.nan, dtype=np.float64)
    np.divide(S, W, out=out, where=W > 0)
    return out


def pairwise(
    a_num: np.ndarray,
    a_cat: np.ndarray,
    b_num: np.ndarray,
    b_cat: np.ndarray,
    w_num: np.ndarray,
    w_cat: np.ndarray,
    *,
    block_elements: int = DEFAULT_BLOCK,
    n_jobs: int = 1,
) -> np.ndarray:
    """Full ``(n_a, n_b)`` Gower distance matrix, computed blockwise."""
    na, nb = a_num.shape[0], b_num.shape[0]
    out = np.empty((na, nb), dtype=np.float64)
    rows = _rows_per_block(max(nb, 1), block_elements)

    def run(start: int) -> None:
        stop = min(start + rows, na)
        S, W = block_sums(
            a_num[start:stop], a_cat[start:stop], b_num, b_cat, w_num, w_cat
        )
        out[start:stop] = _normalise(S, W)

    starts = range(0, na, rows)
    if n_jobs == 1 or na <= rows:
        for start in starts:
            run(start)
    else:
        workers = None if n_jobs < 0 else n_jobs
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, starts))
    return out


def iter_chunks(
    a_num: np.ndarray,
    a_cat: np.ndarray,
    b_num: np.ndarray,
    b_cat: np.ndarray,
    w_num: np.ndarray,
    w_cat: np.ndarray,
    *,
    block_elements: int = DEFAULT_BLOCK,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(start, stop, D_block)`` so callers never hold the full matrix."""
    na, nb = a_num.shape[0], b_num.shape[0]
    rows = _rows_per_block(max(nb, 1), block_elements)
    for start in range(0, na, rows):
        stop = min(start + rows, na)
        S, W = block_sums(a_num[start:stop], a_cat[start:stop], b_num, b_cat, w_num, w_cat)
        yield start, stop, _normalise(S, W)


def topk(
    a_num: np.ndarray,
    a_cat: np.ndarray,
    b_num: np.ndarray,
    b_cat: np.ndarray,
    w_num: np.ndarray,
    w_cat: np.ndarray,
    k: int,
    *,
    exclude_self: bool = False,
    block_elements: int = DEFAULT_BLOCK,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``k`` nearest neighbours without materialising the full matrix.

    Returns ``(distances, indices)``, both ``(n_a, k)`` and sorted ascending.
    ``exclude_self`` drops the ``i == j`` pair, for the common case where the
    query set *is* the reference set.
    """
    na, nb = a_num.shape[0], b_num.shape[0]
    limit = nb - 1 if exclude_self else nb
    if not 1 <= k <= max(limit, 1):
        raise ValueError(f"k must lie in [1, {max(limit, 1)}]; got {k}.")

    dist = np.empty((na, k), dtype=np.float64)
    idx = np.empty((na, k), dtype=np.int64)

    def run(start: int) -> None:
        stop = min(start + rows, na)
        S, W = block_sums(a_num[start:stop], a_cat[start:stop], b_num, b_cat, w_num, w_cat)
        D = _normalise(S, W)
        np.nan_to_num(D, copy=False, nan=np.inf)
        if exclude_self:
            local = np.arange(start, stop)
            D[np.arange(stop - start), local] = np.inf
        part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        taken = np.take_along_axis(D, part, axis=1)
        order = np.argsort(taken, axis=1, kind="stable")
        idx[start:stop] = np.take_along_axis(part, order, axis=1)
        dist[start:stop] = np.take_along_axis(taken, order, axis=1)

    rows = _rows_per_block(max(nb, 1), block_elements)
    starts = range(0, na, rows)
    if n_jobs == 1 or na <= rows:
        for start in starts:
            run(start)
    else:
        workers = None if n_jobs < 0 else n_jobs
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, starts))
    return dist, idx


def column_contributions(
    a_num: np.ndarray,
    a_cat: np.ndarray,
    b_num: np.ndarray,
    b_cat: np.ndarray,
    w_num: np.ndarray,
    w_cat: np.ndarray,
) -> np.ndarray:
    """Per-column contribution to ``d(a_i, b_i)`` for row-aligned inputs.

    ``a`` and ``b`` must have the same number of rows.  Returns an
    ``(n, n_numeric + n_nominal)`` array whose rows sum to the Gower distance,
    which is what makes "why are these two rows similar?" answerable.
    """
    if a_num.shape[0] != b_num.shape[0]:
        raise ValueError("column_contributions requires row-aligned inputs.")
    n = a_num.shape[0]
    parts = np.zeros((n, w_num.size + w_cat.size), dtype=np.float64)
    denom = np.zeros(n, dtype=np.float64)

    for k, w in enumerate(w_num):
        ok = ~(np.isnan(a_num[:, k]) | np.isnan(b_num[:, k]))
        diff = np.zeros(n)
        np.subtract(a_num[:, k], b_num[:, k], out=diff, where=ok)
        parts[:, k] = w * np.abs(diff) * ok
        denom += w * ok

    offset = w_num.size
    for k, w in enumerate(w_cat):
        ok = (a_cat[:, k] >= 0) & (b_cat[:, k] >= 0)
        parts[:, offset + k] = w * ((a_cat[:, k] != b_cat[:, k]) & ok)
        denom += w * ok

    out = np.full_like(parts, np.nan)
    np.divide(parts, denom[:, None], out=out, where=denom[:, None] > 0)
    return out
