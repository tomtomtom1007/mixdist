"""k-prototypes (Huang, 1997) with automatic balancing options.

The cost of a point against a prototype is

.. math::
   d(x, \\mu) = \\sum_{k \\in \\text{num}} w_k (x_k - \\mu_k)^2
              + \\gamma \\sum_{k \\in \\text{cat}} w_k \\mathbb{1}[x_k \\neq \\mu_k],

with numeric columns range-scaled to ``[0, 1]`` by the schema.  The tuning
parameter :math:`\\gamma` is the algorithm's well-known weakness; see the
``gamma`` parameter for the two automatic choices offered here.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEstimator
from ._init import kmeans_plusplus, repair_empty_clusters
from .metric import MixedMetric
from .stats import nanmean, nanstd

__all__ = ["KPrototypes"]


class KPrototypes(BaseEstimator):
    """Cluster mixed-type data with Huang's k-prototypes algorithm.

    Parameters
    ----------
    n_clusters : int, default 8
    gamma : {"auto", "modha-spangler"} or float, default "auto"
        Trade-off between the numeric and categorical terms.

        ``"auto"``
            Huang's rule of thumb: the mean standard deviation of the
            (range-scaled) numeric columns.
        ``"modha-spangler"``
            Search :math:`\\alpha \\in (0, 1)` splitting weight between the two
            blocks and keep the value minimising the product of the numeric and
            categorical within-cluster distortions (Modha & Spangler, 2003).
            Costs one full run per grid point but removes the hand-tuning.
        float
            Use this value directly.
    metric : MixedMetric, optional
        Supplies column typing, scaling and per-column weights.  A fresh
        ``MixedMetric(weights="equal")`` is used when omitted, matching the
        published algorithm.
    n_init : int, default 10
    max_iter : int, default 100
    tol : float, default 0.0
        Stop when the fraction of points changing label falls to ``tol``.
    random_state : int, optional

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
    cost_ : float
        Total within-cluster cost of the retained run.
    gamma_ : float
        The value actually used.
    cluster_centers_ : pandas.DataFrame
        Prototypes in original units: numeric means, categorical modes.

    References
    ----------
    Huang, Z. (1997). *Clustering large data sets with mixed numeric and
    categorical values.* PAKDD.

    Modha, D. S. & Spangler, W. S. (2003). *Feature weighting in k-means
    clustering.* Machine Learning 52(3).
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        gamma="auto",
        metric: MixedMetric | None = None,
        n_init: int = 10,
        max_iter: int = 100,
        tol: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------ fit --
    def fit(self, X, y=None) -> KPrototypes:
        metric = self.metric if self.metric is not None else MixedMetric(weights="equal")
        if not getattr(metric, "_fitted", False):
            metric = metric.fit(X)
        self.metric_ = metric
        num, cat = metric.schema_.encode(X)
        w_num, w_cat = metric.w_num_, metric.w_cat_
        rng = np.random.default_rng(self.random_state)

        if isinstance(self.gamma, str) and self.gamma == "modha-spangler":
            self.gamma_, best = _modha_spangler(
                num, cat, w_num, w_cat, self.n_clusters, self.n_init, self.max_iter, self.tol, rng
            )
        else:
            self.gamma_ = float(_auto_gamma(num, w_num) if self.gamma == "auto" else self.gamma)
            best = _best_of(
                num, cat, w_num, w_cat, self.n_clusters, self.gamma_,
                self.n_init, self.max_iter, self.tol, rng,
            )

        self.labels_, self.cost_, self._centroid_num, self._centroid_cat, self.n_iter_ = best
        self._fitted = True
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X) -> np.ndarray:
        """Assign new rows to the nearest fitted prototype."""
        self._check_fitted()
        num, cat = self.metric_.schema_.encode(X)
        D = _cost_matrix(
            num, cat, self._centroid_num, self._centroid_cat,
            self.metric_.w_num_, self.metric_.w_cat_, self.gamma_,
        )
        return np.asarray(D.argmin(axis=1), dtype=np.int64)

    @property
    def cluster_centers_(self):
        """Prototypes decoded back into the original units and level names."""
        self._check_fitted()
        return _decode_centres(self.metric_.schema_, self._centroid_num, self._centroid_cat)


# --------------------------------------------------------------------------- #
def _auto_gamma(num: np.ndarray, w_num: np.ndarray) -> float:
    if num.shape[1] == 0:
        return 1.0
    sds = nanstd(num, axis=0)
    sds = sds[np.isfinite(sds)]
    value = float(np.mean(sds)) if sds.size else 0.0
    return value if value > 0 else 1.0


def _cost_matrix(num, cat, c_num, c_cat, w_num, w_cat, gamma: float) -> np.ndarray:
    """``(n, k)`` cost of every point against every prototype."""
    n, k = num.shape[0], c_num.shape[0]
    D = np.zeros((n, k), dtype=np.float64)
    for j, w in enumerate(w_num):
        if w == 0.0:
            continue
        diff = num[:, j][:, None] - c_num[None, :, j]
        np.nan_to_num(diff, copy=False, nan=0.0)
        D += w * diff * diff
    for j, w in enumerate(w_cat):
        if w == 0.0:
            continue
        col = cat[:, j][:, None]
        D += gamma * w * ((col != c_cat[None, :, j]) & (col >= 0))
    return D


def _update_centres(num, cat, labels, k):
    c_num = np.zeros((k, num.shape[1]), dtype=np.float64)
    c_cat = np.zeros((k, cat.shape[1]), dtype=np.int32)
    for g in range(k):
        mask = labels == g
        if not np.any(mask):  # pragma: no cover - guarded by repair_empty_clusters
            continue
        if num.shape[1]:
            means = nanmean(num[mask], axis=0)
            c_num[g] = np.where(np.isnan(means), 0.5, means)
        for j in range(cat.shape[1]):
            codes = cat[mask, j]
            codes = codes[codes >= 0]
            c_cat[g, j] = np.bincount(codes).argmax() if codes.size else 0
    return c_num, c_cat


def _run_once(num, cat, w_num, w_cat, k, gamma, max_iter, tol, rng):
    n = num.shape[0]

    def dist_to_point(i: int) -> np.ndarray:
        return _cost_matrix(
            num, cat, num[i : i + 1], cat[i : i + 1], w_num, w_cat, gamma
        ).ravel()

    seeds = kmeans_plusplus(n, k, rng, dist_to_point)
    c_num, c_cat = num[seeds].copy(), cat[seeds].copy()
    np.nan_to_num(c_num, copy=False, nan=0.5)

    labels = np.full(n, -1, dtype=np.int64)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - read after the loop
        D = _cost_matrix(num, cat, c_num, c_cat, w_num, w_cat, gamma)
        new_labels = np.asarray(D.argmin(axis=1), dtype=np.int64)
        new_labels = repair_empty_clusters(new_labels, D, k, rng)
        moved = float(np.mean(new_labels != labels))
        labels = new_labels
        c_num, c_cat = _update_centres(num, cat, labels, k)
        if moved <= tol:
            break

    D = _cost_matrix(num, cat, c_num, c_cat, w_num, w_cat, gamma)
    cost = float(D[np.arange(n), labels].sum())
    return labels, cost, c_num, c_cat, n_iter


def _best_of(num, cat, w_num, w_cat, k, gamma, n_init, max_iter, tol, rng):
    best = None
    for _ in range(max(int(n_init), 1)):
        result = _run_once(num, cat, w_num, w_cat, k, gamma, max_iter, tol, rng)
        if best is None or result[1] < best[1]:
            best = result
    return best


def _domain_distortions(num, cat, labels, c_num, c_cat, w_num, w_cat):
    """Within-cluster distortion measured separately in each domain."""
    n = num.shape[0]
    idx = np.arange(n)
    num_cost = _cost_matrix(
        num, cat, c_num, c_cat, w_num, np.zeros_like(w_cat), 1.0
    )[idx, labels].sum()
    cat_cost = _cost_matrix(
        num, cat, c_num, c_cat, np.zeros_like(w_num), w_cat, 1.0
    )[idx, labels].sum()
    return float(num_cost), float(cat_cost)


def _modha_spangler(num, cat, w_num, w_cat, k, n_init, max_iter, tol, rng, n_grid: int = 9):
    """Pick the numeric/categorical balance by minimising the distortion product.

    Modha & Spangler's criterion needs no labels and no tuning: the optimum
    trades the two domains off against each other rather than letting whichever
    block happens to have more variance win.
    """
    if num.shape[1] == 0 or cat.shape[1] == 0:
        gamma = _auto_gamma(num, w_num)
        return gamma, _best_of(num, cat, w_num, w_cat, k, gamma, n_init, max_iter, tol, rng)

    alphas = np.linspace(1.0, n_grid, n_grid) / (n_grid + 1.0)
    best_score, best_gamma, best_run = np.inf, None, None
    for alpha in alphas:
        gamma = float((1.0 - alpha) / alpha)
        run = _best_of(num, cat, w_num, w_cat, k, gamma, max(1, n_init // 2), max_iter, tol, rng)
        labels, _, c_num, c_cat, _ = run
        d_num, d_cat = _domain_distortions(num, cat, labels, c_num, c_cat, w_num, w_cat)
        score = d_num * d_cat
        if score < best_score:
            best_score, best_gamma, best_run = score, gamma, run
    return best_gamma, best_run


def _decode_centres(schema, c_num, c_cat):
    import pandas as pd

    data = {}
    for col in schema.numeric:
        data[col.name] = col.decode_scaled(c_num[:, col.slot])
    for col in schema.nominal:
        cats = col.categories
        codes = np.clip(c_cat[:, col.slot], 0, max(len(cats) - 1, 0))
        data[col.name] = np.asarray(cats)[codes]
    return pd.DataFrame(data, index=pd.RangeIndex(c_num.shape[0], name="cluster"))
