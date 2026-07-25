"""``MixedMetric`` — the entry point of the library."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ._base import BaseEstimator
from .distance import (
    DEFAULT_BLOCK,
    column_contributions,
    iter_chunks,
    pairwise,
    topk,
)
from .embedding import ThermometerEmbedding, impute_for_embedding
from .schema import Schema
from .stats import expected_dissimilarity, nanmedian
from .weighting import resolve_weights

__all__ = ["MixedMetric"]


class MixedMetric(BaseEstimator):
    """A fitted, inspectable distance over mixed numeric/categorical tables.

    Parameters
    ----------
    weights : {"balanced", "equal", "type_balanced"}, mapping or sequence, default "balanced"
        How much each column contributes.  ``"equal"`` is Gower (1971) as
        published; ``"balanced"`` equalises each column's *expected*
        contribution and is the default because equal nominal weights let
        high-cardinality categoricals dominate.  See :mod:`mixdist.weighting`.
    kinds, categorical, ordinal : optional
        Column typing overrides passed through to :meth:`mixdist.Schema.fit`.
    numeric_range : {"minmax", "robust"}, default "minmax"
        Scaling range for numeric columns.  ``"robust"`` uses inner quantiles
        and clips, so one outlier cannot flatten a variable.
    clip_quantile : float, default 0.01
        Tail probability for ``numeric_range="robust"``.
    drop_constant : bool, default True
        Drop single-valued columns instead of letting them rescale distances.
    n_bins : int, default 128
        Thermometer resolution used by :meth:`transform`.
    n_jobs : int, default 1
        Threads for blockwise computation; ``-1`` uses the default pool size.
    block_elements : int, default 8_000_000
        Elements per distance block, i.e. the memory knob.

    Attributes
    ----------
    schema_ : Schema
    weights_ : pandas.Series
        Final per-column weights, indexed by column name.

    Examples
    --------
    >>> from mixdist import MixedMetric, make_mixed_blobs
    >>> X, y = make_mixed_blobs(n_samples=200, random_state=0)
    >>> metric = MixedMetric().fit(X)
    >>> D = metric.pairwise(X)
    >>> D.shape
    (200, 200)
    """

    def __init__(
        self,
        weights="balanced",
        *,
        kinds=None,
        categorical=None,
        ordinal=None,
        numeric_range: str = "minmax",
        clip_quantile: float = 0.01,
        drop_constant: bool = True,
        n_bins: int = 128,
        n_jobs: int = 1,
        block_elements: int = DEFAULT_BLOCK,
    ) -> None:
        self.weights = weights
        self.kinds = kinds
        self.categorical = categorical
        self.ordinal = ordinal
        self.numeric_range = numeric_range
        self.clip_quantile = clip_quantile
        self.drop_constant = drop_constant
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.block_elements = block_elements

    # ------------------------------------------------------------------ fit --
    def fit(self, X, y=None) -> MixedMetric:
        """Learn column types, scaling bounds and weights from ``X``."""
        self.schema_ = Schema.fit(
            X,
            kinds=self.kinds,
            categorical=self.categorical,
            ordinal=self.ordinal,
            numeric_range=self.numeric_range,
            clip_quantile=self.clip_quantile,
            drop_constant=self.drop_constant,
        )
        num, cat = self.schema_.encode(X)
        self.w_num_, self.w_cat_ = resolve_weights(self.weights, self.schema_, num, cat)
        self.expected_num_, self.expected_cat_ = expected_dissimilarity(num, cat)
        fill = nanmedian(num, axis=0) if num.shape[1] else np.zeros(0)
        self.numeric_fill_ = np.where(np.isnan(fill), 0.5, fill)
        self._num, self._cat = num, cat
        self._fitted = True
        return self

    def fit_transform(self, X, y=None, **kwargs) -> np.ndarray:
        return self.fit(X).transform(X, **kwargs)

    # ------------------------------------------------------------ inspection -
    @property
    def weights_(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(
            np.concatenate([self.w_num_, self.w_cat_]),
            index=self.schema_.names,
            name="weight",
        )

    def column_report(self) -> pd.DataFrame:
        """Per-column table of type, dispersion, weight and share of the distance.

        ``share`` is :math:`w_k E[d_k] / \\sum_j w_j E[d_j]`: the fraction of a
        typical pairwise distance that the column actually commands.  Reading
        this before trusting a clustering is the single highest-value habit
        this library tries to encourage — under ``weights="equal"`` it is
        common to find one categorical column holding 40 % of the distance.
        """
        self._check_fitted()
        cols = self.schema_.numeric + self.schema_.nominal
        weight = np.concatenate([self.w_num_, self.w_cat_])
        expected = np.concatenate([self.expected_num_, self.expected_cat_])
        mass = weight * expected
        total = mass.sum()
        return pd.DataFrame(
            {
                "kind": [c.kind for c in cols],
                "n_levels": [c.n_levels if not c.is_numeric else np.nan for c in cols],
                "expected_dissimilarity": expected,
                "weight": weight,
                "share": mass / total if total > 0 else np.zeros_like(mass),
            },
            index=pd.Index([c.name for c in cols], name="column"),
        ).sort_values("share", ascending=False)

    # -------------------------------------------------------------- distances -
    def pairwise(self, X=None, Y=None) -> np.ndarray:
        """Full ``(n_X, n_Y)`` distance matrix.  ``Y=None`` means ``Y = X``.

        Memory is ``O(n_X * n_Y)`` by definition; use :meth:`kneighbors` or
        :meth:`iter_pairwise` when that does not fit.
        """
        a_num, a_cat = self._encode(X)
        b_num, b_cat = (a_num, a_cat) if Y is None else self._encode(Y)
        return pairwise(
            a_num,
            a_cat,
            b_num,
            b_cat,
            self.w_num_,
            self.w_cat_,
            block_elements=self.block_elements,
            n_jobs=self.n_jobs,
        )

    def iter_pairwise(self, X=None, Y=None) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield ``(start, stop, block)`` slices of the distance matrix."""
        a_num, a_cat = self._encode(X)
        b_num, b_cat = (a_num, a_cat) if Y is None else self._encode(Y)
        return iter_chunks(
            a_num, a_cat, b_num, b_cat, self.w_num_, self.w_cat_,
            block_elements=self.block_elements,
        )

    def kneighbors(self, X=None, Y=None, n_neighbors: int = 5, *, exclude_self=None):
        """Exact ``k`` nearest neighbours under the Gower geometry.

        Never materialises the full matrix, so this is the method to reach for
        on large tables.  Returns ``(distances, indices)``, both sorted
        ascending.  ``exclude_self`` defaults to ``True`` when ``Y is None``.
        """
        a_num, a_cat = self._encode(X)
        if Y is None:
            b_num, b_cat = a_num, a_cat
            drop_self = True if exclude_self is None else bool(exclude_self)
        else:
            b_num, b_cat = self._encode(Y)
            drop_self = False if exclude_self is None else bool(exclude_self)
        return topk(
            a_num,
            a_cat,
            b_num,
            b_cat,
            self.w_num_,
            self.w_cat_,
            n_neighbors,
            exclude_self=drop_self,
            block_elements=self.block_elements,
            n_jobs=self.n_jobs,
        )

    # -------------------------------------------------------------- embedding -
    def embedding(self, *, n_components=None, random_state=None, dtype=np.float32):
        """Build the :class:`~mixdist.embedding.ThermometerEmbedding` for this metric."""
        self._check_fitted()
        return ThermometerEmbedding(
            numeric_names=[c.name for c in self.schema_.numeric],
            nominal_names=[c.name for c in self.schema_.nominal],
            n_levels=[c.n_levels for c in self.schema_.nominal],
            w_num=self.w_num_,
            w_cat=self.w_cat_,
            n_bins=self.n_bins,
            n_components=n_components,
            random_state=random_state,
            dtype=dtype,
        )

    def transform(
        self,
        X=None,
        *,
        n_components=None,
        random_state=None,
        on_missing: str = "impute",
        dtype=np.float32,
    ) -> np.ndarray:
        """Embed ``X`` so that **squared** Euclidean distance equals this metric.

        ``||transform(X)[i] - transform(X)[j]||^2 == pairwise(X)[i, j]``, up to
        ``1 / n_bins`` quantisation per numeric column.  Feed the result to
        FAISS, hnswlib, UMAP or ``KMeans`` to get Gower geometry out of
        Euclidean tooling.  See :mod:`mixdist.embedding` for the construction.
        """
        num, cat = self._encode(X)
        num, cat = impute_for_embedding(
            num, cat, numeric_fill=self.numeric_fill_, on_missing=on_missing
        )
        emb = self.embedding(n_components=n_components, random_state=random_state, dtype=dtype)
        return emb.transform(num, cat)

    # ------------------------------------------------------------- explanation -
    def explain_pairs(self, A, B) -> pd.DataFrame:
        """Per-column contributions for row-aligned ``A`` and ``B``.

        Each row sums to the Gower distance between the corresponding rows, so
        the decomposition is exact and additive — no surrogate model involved.
        """
        a_num, a_cat = self._encode(A)
        b_num, b_cat = self._encode(B)
        parts = column_contributions(
            a_num, a_cat, b_num, b_cat, self.w_num_, self.w_cat_
        )
        return pd.DataFrame(parts, columns=self.schema_.names)

    def explain(self, X, i: int, j: int) -> pd.Series:
        """Why rows ``i`` and ``j`` of ``X`` are (dis)similar, column by column."""
        frame = self._as_frame(X)
        pair = self.explain_pairs(frame.iloc[[i]], frame.iloc[[j]]).iloc[0]
        pair.name = f"d(row {i}, row {j}) = {pair.sum():.4f}"
        return pair.sort_values(ascending=False)

    def explain_clusters(self, X, labels) -> pd.DataFrame:
        """Which variables actually define each cluster.

        Entry ``(g, k)`` is the weighted **dispersion reduction** of column
        ``k`` inside cluster ``g``,

        .. math:: \\frac{w_k}{W}\\bigl(E[d_k] - E[d_k \\mid \\text{cluster } g]\\bigr),

        i.e. how much tighter the column becomes once you condition on the
        cluster, expressed on the Gower scale so the row sums are comparable to
        distances.  A column that is homogeneous within the cluster scores
        high; a column that is just as scattered inside the cluster as outside
        scores ~0 **however far it sits from the global centre** — which is why
        this is measured rather than the distance to a central row.  Negative
        entries mean the cluster is *more* heterogeneous than the data at large.
        """
        self._check_fitted()
        frame = self._as_frame(X)
        labels = np.asarray(labels)
        if labels.shape[0] != len(frame):
            raise ValueError("labels must have one entry per row of X.")
        num, cat = self.schema_.encode(frame)

        overall = np.concatenate(expected_dissimilarity(num, cat))
        weights = np.concatenate([self.w_num_, self.w_cat_])
        total = float(weights.sum())

        rows = {}
        for g in np.unique(labels[labels >= 0]):
            mask = labels == g
            within = np.concatenate(expected_dissimilarity(num[mask], cat[mask]))
            rows[g] = weights * (overall - within) / total
        return pd.DataFrame(rows, index=self.schema_.names).T.rename_axis("cluster")

    # ---------------------------------------------------------------- private -
    def _encode(self, X):
        self._check_fitted()
        if X is None:
            return self._num, self._cat
        return self.schema_.encode(X)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        from .schema import as_frame

        return as_frame(X)
