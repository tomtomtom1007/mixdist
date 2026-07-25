r"""Exact Euclidean embedding of the Gower distance.

Why this is possible
--------------------
Gower's distance is a non-negatively weighted average of two elementary
dissimilarities:

* :math:`|u - v|` on a range-scaled numeric variable — the :math:`L_1` metric
  on the line;
* :math:`\mathbb{1}[u \neq v]` on a nominal variable — half the :math:`L_1`
  metric between one-hot vectors.

Both are metrics *of negative type*, and negative type is closed under
non-negative linear combination.  Hence :math:`\sqrt{d_{\text{Gower}}}` embeds
isometrically into :math:`\ell_2`, and the embedding can be written down
explicitly rather than recovered numerically by MDS.

The map
-------
For numeric column :math:`k` with weight :math:`w_k`, total weight :math:`W`
and :math:`m` bins, emit :math:`m` "thermometer" features

.. math:: \phi_l(u) = a_k \,\mathbb{1}\!\left[u > \tfrac{l + 0.5}{m}\right],
          \qquad a_k = \sqrt{w_k / (W m)},

so that :math:`\|\phi(u) - \phi(v)\|^2 = a_k^2 \lvert c(u) - c(v) \rvert
\approx w_k |u - v| / W`, with quantisation error at most :math:`w_k / (W m)`.

For nominal column :math:`k`, emit one-hot features scaled by
:math:`b_k = \sqrt{w_k / (2W)}`, giving :math:`\|\cdot\|^2 = w_k / W` exactly
when the levels differ and :math:`0` when they agree.

Summing over columns yields

.. math:: \|z_i - z_j\|_2^2 = d_{\text{Gower}}(i, j).

Consequences
------------
Squared Euclidean distance in the embedded space *is* the Gower distance, so
any Euclidean tool — FAISS, hnswlib, ``KMeans``, UMAP — operates on Gower
geometry without approximation, and nearest-neighbour rankings are identical
(``L2`` is a monotone transform of ``L2**2``).
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["ThermometerEmbedding"]


class ThermometerEmbedding:
    """Builds the explicit feature map described in the module docstring.

    Constructed by :meth:`mixdist.MixedMetric.embedding`; rarely instantiated
    directly.
    """

    def __init__(
        self,
        *,
        numeric_names: list[str],
        nominal_names: list[str],
        n_levels: list[int],
        w_num: np.ndarray,
        w_cat: np.ndarray,
        n_bins: int = 128,
        n_components: int | None = None,
        random_state: int | None = None,
        dtype=np.float32,
    ) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1.")
        self.numeric_names = list(numeric_names)
        self.nominal_names = list(nominal_names)
        self.n_levels = list(n_levels)
        self.w_num = np.asarray(w_num, dtype=np.float64)
        self.w_cat = np.asarray(w_cat, dtype=np.float64)
        self.n_bins = int(n_bins)
        self.n_components = n_components
        self.random_state = random_state
        self.dtype = dtype

        total = float(self.w_num.sum() + self.w_cat.sum())
        if total <= 0:
            raise ValueError("Total weight must be positive.")
        self._amp_num = np.sqrt(self.w_num / (total * self.n_bins))
        self._amp_cat = np.sqrt(self.w_cat / (2.0 * total))

        self.exact_n_features_ = self.n_bins * len(self.numeric_names) + int(sum(self.n_levels))
        self._projection: np.ndarray | None = None
        if n_components is not None:
            if n_components < 1:
                raise ValueError("n_components must be >= 1.")
            if n_components < self.exact_n_features_:
                rng = np.random.default_rng(random_state)
                self._projection = rng.standard_normal(
                    (self.exact_n_features_, int(n_components))
                ) / np.sqrt(n_components)

    # ------------------------------------------------------------------ API --
    @property
    def n_features_out_(self) -> int:
        return self.exact_n_features_ if self._projection is None else self._projection.shape[1]

    @property
    def is_exact(self) -> bool:
        """``True`` when no random projection is applied (numeric quantisation aside)."""
        return self._projection is None

    def feature_names_out(self) -> list[str]:
        """Names of the exact features; unavailable once randomly projected."""
        if self._projection is not None:
            return [f"rp{i}" for i in range(self.n_features_out_)]
        names: list[str] = []
        for name in self.numeric_names:
            names += [f"{name}>bin{b}" for b in range(self.n_bins)]
        for name, levels in zip(self.nominal_names, self.n_levels):
            names += [f"{name}=level{v}" for v in range(levels)]
        return names

    def transform(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Map encoded ``(num, cat)`` arrays to the embedded space."""
        n = num.shape[0]
        Z = np.zeros((n, self.exact_n_features_), dtype=np.float64)
        offset = 0

        ladder = np.arange(self.n_bins, dtype=np.float64)[None, :]
        for k in range(len(self.numeric_names)):
            u = num[:, k]
            counts = np.clip(np.ceil(self.n_bins * u - 0.5), 0.0, self.n_bins)
            Z[:, offset : offset + self.n_bins] = self._amp_num[k] * (ladder < counts[:, None])
            offset += self.n_bins

        for k, levels in enumerate(self.n_levels):
            codes = cat[:, k]
            valid = codes >= 0
            rows = np.flatnonzero(valid)
            Z[rows, offset + codes[rows]] = self._amp_cat[k]
            offset += levels

        if self._projection is not None:
            Z = Z @ self._projection
        return Z.astype(self.dtype, copy=False)


def impute_for_embedding(
    num: np.ndarray,
    cat: np.ndarray,
    *,
    numeric_fill: np.ndarray,
    on_missing: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve missing values so the embedding is well defined.

    Gower handles a missing value by *removing the column from that pair's
    denominator*, which makes the effective metric pair-dependent — and a
    pair-dependent metric has no single feature map.  There is no exact fix, so
    the behaviour is explicit:

    ``on_missing="error"``
        Refuse to embed.
    ``on_missing="impute"``
        Numeric gaps take the column median; nominal gaps are emitted as an
        all-zero block, which places them at distance ``sqrt(w_k / 2W)`` from
        every observed level rather than Gower's ``sqrt(w_k / W)``.
    """
    n_missing_num = int(np.isnan(num).sum())
    n_missing_cat = int((cat < 0).sum())
    if n_missing_num == 0 and n_missing_cat == 0:
        return num, cat

    if on_missing == "error":
        raise ValueError(
            f"Cannot embed data with missing values ({n_missing_num} numeric, "
            f"{n_missing_cat} categorical): Gower's pairwise denominator makes the "
            "metric pair-dependent. Pass on_missing='impute' to accept the "
            "documented approximation, or impute upstream."
        )
    if on_missing != "impute":
        raise ValueError("on_missing must be 'error' or 'impute'.")

    warnings.warn(
        f"Imputing {n_missing_num} numeric and {n_missing_cat} categorical missing "
        "values for the embedding; embedded distances will differ from pairwise() "
        "for the affected rows. See mixdist.embedding.impute_for_embedding.",
        UserWarning,
        stacklevel=3,
    )
    num = np.where(np.isnan(num), numeric_fill[None, :], num)
    return num, cat
