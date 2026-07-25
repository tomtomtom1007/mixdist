"""KAMILA — KAy-means for MIxed LArge data sets.

A semiparametric alternative to k-prototypes whose selling point is that it has
**no** numeric/categorical trade-off parameter to tune.  Instead of adding two
incommensurable distances, it compares two *likelihoods*:

* the continuous block is modelled as a spherical cluster whose radial density
  :math:`\\hat f_r` is estimated non-parametrically by KDE on the
  point-to-nearest-centroid distances, giving a log-density
  :math:`\\log \\hat f_r(r) - (p - 1)\\log r` for a :math:`p`-dimensional
  spherically symmetric law;
* the categorical block is modelled as independent multinomials per cluster.

Because both terms are log-densities they are already on a common scale, and
the balance between them is set by the data rather than by a :math:`\\gamma`.
That is precisely the failure mode of Gower and k-prototypes that this library
exists to address.

.. note::
   Implemented from the published description (Foss, Markatou, Hunter &
   Richardson, 2016), not ported from the reference R package ``kamila``.
   Results agree in behaviour but need not match it bit for bit.

References
----------
Foss, A., Markatou, M., Ray, B. & Heching, A. (2016). *A semiparametric method
for clustering mixed data.* Machine Learning 105(3), 419-458.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEstimator
from ._init import kmeans_plusplus, repair_empty_clusters
from .metric import MixedMetric
from .stats import nanmean, nanstd

__all__ = ["KAMILA"]

_TINY = 1e-300


class KAMILA(BaseEstimator):
    """Cluster mixed-type data without a numeric/categorical weight parameter.

    Parameters
    ----------
    n_clusters : int, default 8
    metric : MixedMetric, optional
        Supplies column typing and scaling.  Defaults to
        ``MixedMetric(weights="equal")`` — KAMILA derives the between-block
        balance itself, so pre-balancing the blocks is redundant.  Per-column
        weights, if given, still act *within* each block.
    n_init : int, default 10
    max_iter : int, default 50
    tol : float, default 0.0
        Stop when the fraction of points changing label falls to ``tol``.
    smoothing : float, default 1.0
        Additive smoothing for the multinomial level probabilities.  Keeps an
        unobserved level from producing ``log 0``.
    bandwidth : {"silverman"} or float, default "silverman"
        Bandwidth for the radial KDE.
    n_grid : int, default 512
        Grid resolution at which the KDE is evaluated before interpolation;
        this is what keeps the estimator ``O(n)`` rather than ``O(n^2)``.
    random_state : int, optional

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
    log_likelihood_ : float
        Classification log-likelihood of the retained run (higher is better).
    cluster_centers_ : pandas.DataFrame
        Continuous means (original units) and modal categorical levels.
    level_probabilities_ : dict[str, pandas.DataFrame]
        Per-cluster level probabilities for each categorical column — the
        directly readable description of what separates the clusters.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        metric: MixedMetric | None = None,
        n_init: int = 10,
        max_iter: int = 50,
        tol: float = 0.0,
        smoothing: float = 1.0,
        bandwidth="silverman",
        n_grid: int = 512,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.bandwidth = bandwidth
        self.n_grid = n_grid
        self.random_state = random_state

    # ------------------------------------------------------------------ fit --
    def fit(self, X, y=None) -> KAMILA:
        metric = self.metric if self.metric is not None else MixedMetric(weights="equal")
        if not getattr(metric, "_fitted", False):
            metric = metric.fit(X)
        self.metric_ = metric

        num, cat = metric.schema_.encode(X)
        V, self._centre, self._scale = _standardise(num, metric.w_num_)
        levels = np.array([c.n_levels for c in metric.schema_.nominal], dtype=np.int64)
        w_cat = metric.w_cat_
        self._w_cat = w_cat  # needed by _scores during the runs below
        self._levels = levels
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(max(int(self.n_init), 1)):
            result = self._run_once(V, cat, levels, w_cat, rng)
            if best is None or result["loglik"] > best["loglik"]:
                best = result

        self.labels_ = best["labels"]
        self.log_likelihood_ = best["loglik"]
        self.n_iter_ = best["n_iter"]
        self._mu = best["mu"]
        self._theta = best["theta"]
        self._kde_grid = best["kde_grid"]
        self._kde_dens = best["kde_dens"]
        self._fitted = True
        return self

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X) -> np.ndarray:
        """Assign new rows using the fitted centroids, KDE and multinomials."""
        self._check_fitted()
        num, cat = self.metric_.schema_.encode(X)
        V = _apply_standardise(num, self._centre, self._scale)
        scores = self._scores(V, cat, self._mu, self._theta, self._kde_grid, self._kde_dens)
        return np.asarray(scores.argmax(axis=1), dtype=np.int64)

    # -------------------------------------------------------------- reporting -
    @property
    def cluster_centers_(self):
        self._check_fitted()
        import pandas as pd

        schema = self.metric_.schema_
        data = {}
        scaled = _invert_standardise(self._mu, self._centre, self._scale)
        for col in schema.numeric:
            data[col.name] = col.decode_scaled(scaled[:, col.slot])
        for col in schema.nominal:
            probs = self._theta[col.slot]
            data[col.name] = np.asarray(col.categories)[probs.argmax(axis=1)]
        return pd.DataFrame(data, index=pd.RangeIndex(self.n_clusters, name="cluster"))

    @property
    def level_probabilities_(self):
        self._check_fitted()
        import pandas as pd

        out = {}
        for col in self.metric_.schema_.nominal:
            out[col.name] = pd.DataFrame(
                self._theta[col.slot],
                index=pd.RangeIndex(self.n_clusters, name="cluster"),
                columns=[str(c) for c in col.categories],
            )
        return out

    # ---------------------------------------------------------------- internals
    def _run_once(self, V, cat, levels, w_cat, rng):
        n, p = V.shape
        k = int(self.n_clusters)

        if p:
            seeds = kmeans_plusplus(
                n, k, rng, lambda i: np.linalg.norm(V - V[i], axis=1)
            )
            mu = V[seeds].copy()
        else:
            mu = np.zeros((k, 0))
        labels = rng.integers(0, k, size=n)
        theta = _fit_theta(cat, labels, levels, k, self.smoothing)
        grid = dens = None

        n_iter = 0
        for n_iter in range(1, int(self.max_iter) + 1):  # noqa: B007 - read after the loop
            grid, dens = self._fit_radial_kde(V, mu)
            scores = self._scores(V, cat, mu, theta, grid, dens)
            new_labels = np.asarray(scores.argmax(axis=1), dtype=np.int64)
            new_labels = repair_empty_clusters(new_labels, -scores, k, rng)
            moved = float(np.mean(new_labels != labels))
            labels = new_labels
            mu = _update_mu(V, labels, k)
            theta = _fit_theta(cat, labels, levels, k, self.smoothing)
            if moved <= self.tol:
                break

        grid, dens = self._fit_radial_kde(V, mu)
        scores = self._scores(V, cat, mu, theta, grid, dens)
        loglik = float(scores[np.arange(n), labels].sum())
        return {
            "labels": labels,
            "loglik": loglik,
            "mu": mu,
            "theta": theta,
            "kde_grid": grid,
            "kde_dens": dens,
            "n_iter": n_iter,
        }

    def _fit_radial_kde(self, V, mu):
        """KDE of the radial density, fitted on point-to-nearest-centroid distances."""
        if V.shape[1] == 0:
            return None, None
        radii = _radii(V, mu)
        rmin = radii.min(axis=1)
        h = _bandwidth(rmin, self.bandwidth)
        upper = float(radii.max()) + 4.0 * h
        grid = np.linspace(0.0, max(upper, h), int(self.n_grid))
        dens = _kde_on_grid(rmin, grid, h)
        return grid, dens

    def _scores(self, V, cat, mu, theta, grid, dens):
        n, p = V.shape
        k = mu.shape[0]
        scores = np.zeros((n, k), dtype=np.float64)

        if p:
            radii = _radii(V, mu)
            eps = max(float(np.median(radii)) * 1e-6, 1e-12)
            safe = np.maximum(radii, eps)
            f = np.interp(safe, grid, dens, left=dens[0], right=_TINY)
            scores += np.log(np.maximum(f, _TINY))
            if p > 1:
                scores -= (p - 1) * np.log(safe)

        for j, probs in enumerate(theta):
            codes = cat[:, j]
            valid = codes >= 0
            logp = np.log(np.maximum(probs, _TINY))  # (k, levels)
            contrib = np.zeros((n, k))
            contrib[valid] = logp[:, codes[valid]].T
            scores += float(self._w_cat[j]) * contrib
        return scores


# --------------------------------------------------------------------------- #
def _standardise(num: np.ndarray, w_num: np.ndarray):
    """Centre, scale to unit variance, then apply within-block weights."""
    if num.shape[1] == 0:
        return np.zeros((num.shape[0], 0)), np.zeros(0), np.ones(0)
    centre = nanmean(num, axis=0)
    spread = nanstd(num, axis=0)
    centre = np.where(np.isnan(centre), 0.0, centre)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    mean_w = float(np.mean(w_num)) if w_num.size else 1.0
    gain = np.sqrt(w_num / mean_w) if mean_w > 0 else np.ones_like(w_num)
    scale = spread / np.where(gain > 0, gain, 1.0)
    return _apply_standardise(num, centre, scale), centre, scale


def _apply_standardise(num, centre, scale):
    if num.shape[1] == 0:
        return np.zeros((num.shape[0], 0))
    V = (num - centre) / scale
    return np.nan_to_num(V, nan=0.0)  # missing sits at the column mean


def _invert_standardise(mu, centre, scale):
    if mu.shape[1] == 0:
        return mu
    return mu * scale + centre


def _radii(V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``(n, k)`` Euclidean distances, computed without an ``(n, k, p)`` temporary."""
    sq = (
        np.einsum("ij,ij->i", V, V)[:, None]
        - 2.0 * (V @ mu.T)
        + np.einsum("ij,ij->i", mu, mu)[None, :]
    )
    return np.sqrt(np.maximum(sq, 0.0))


def _update_mu(V, labels, k):
    if V.shape[1] == 0:
        return np.zeros((k, 0))
    mu = np.zeros((k, V.shape[1]))
    for g in range(k):
        mask = labels == g
        if np.any(mask):
            mu[g] = V[mask].mean(axis=0)
    return mu


def _fit_theta(cat, labels, levels, k, smoothing):
    """Per-cluster multinomial level probabilities with additive smoothing."""
    theta = []
    for j, n_levels in enumerate(levels):
        probs = np.empty((k, int(n_levels)), dtype=np.float64)
        for g in range(k):
            codes = cat[labels == g, j]
            codes = codes[codes >= 0]
            counts = np.bincount(codes, minlength=int(n_levels)).astype(np.float64)
            probs[g] = (counts + smoothing) / (counts.sum() + smoothing * n_levels)
        theta.append(probs)
    return theta


def _bandwidth(samples: np.ndarray, rule) -> float:
    if not isinstance(rule, str):
        value = float(rule)
        if value <= 0:
            raise ValueError("bandwidth must be positive.")
        return value
    if rule != "silverman":
        raise ValueError("bandwidth must be 'silverman' or a positive float.")
    n = samples.size
    if n < 2:
        return 1.0
    sd = float(np.std(samples))
    q1, q3 = np.quantile(samples, [0.25, 0.75])
    spread = min(sd, (q3 - q1) / 1.34) if q3 > q1 else sd
    h = 0.9 * spread * n ** (-0.2)
    if h > 0:
        return float(h)
    return float(sd * n ** (-0.2)) if sd > 0 else 1e-3


def _kde_on_grid(samples: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    """Gaussian KDE with reflection at zero (radii are non-negative)."""
    dens = np.zeros_like(grid)
    chunk = max(1, int(2_000_000 // max(grid.size, 1)))
    for start in range(0, samples.size, chunk):
        block = samples[start : start + chunk]
        z = (grid[:, None] - block[None, :]) / h
        dens += np.exp(-0.5 * z * z).sum(axis=1)
        z_ref = (grid[:, None] + block[None, :]) / h
        dens += np.exp(-0.5 * z_ref * z_ref).sum(axis=1)
    return dens / (samples.size * h * np.sqrt(2.0 * np.pi))
