"""mixdist — distances, neighbour search and clustering for mixed-type tables.

Quick start
-----------
>>> from mixdist import MixedMetric, make_mixed_blobs
>>> X, y = make_mixed_blobs(n_samples=300, random_state=0)
>>> metric = MixedMetric(weights="balanced").fit(X)
>>> metric.column_report().head(3)            # doctest: +SKIP
>>> dist, idx = metric.kneighbors(n_neighbors=5)   # never builds the full matrix
>>> Z = metric.transform(X)                   # ||Z_i - Z_j||^2 == Gower distance

The three things this package tries to do better than the alternatives:

1. **Weighting is explicit and inspectable.**  ``column_report()`` shows the
   share of the distance each column actually commands, and ``"balanced"``
   equalises those shares instead of the nominal weights.
2. **It scales.**  Distances accumulate one column at a time into a bounded
   block, and :meth:`MixedMetric.kneighbors` never materialises ``n x n``.
3. **It plugs into Euclidean tooling.**  :meth:`MixedMetric.transform` is an
   exact feature map for the Gower geometry, so FAISS, hnswlib, UMAP and
   ``KMeans`` all work unmodified.
"""

from __future__ import annotations

from ._base import NotFittedError
from .datasets import make_mixed_blobs
from .embedding import ThermometerEmbedding
from .kamila import KAMILA
from .kprototypes import KPrototypes
from .metric import MixedMetric
from .schema import Column, Schema, infer_kinds
from .stats import expected_dissimilarity, gini_impurity, gini_mean_difference

__version__ = "0.1.0"

__all__ = [
    "KAMILA",
    "Column",
    "KPrototypes",
    "MixedMetric",
    "NotFittedError",
    "Schema",
    "ThermometerEmbedding",
    "expected_dissimilarity",
    "gini_impurity",
    "gini_mean_difference",
    "gower_matrix",
    "infer_kinds",
    "make_mixed_blobs",
    "__version__",
]


def gower_matrix(X, Y=None, *, weights="equal", **kwargs):
    """Gower distance matrix in one call.

    A drop-in replacement for the common ``gower.gower_matrix`` idiom.  The
    default ``weights="equal"`` reproduces Gower (1971) exactly; pass
    ``weights="balanced"`` to stop high-cardinality categoricals from
    dominating.  For anything you will reuse, fit a :class:`MixedMetric`
    instead — it keeps the schema, the weights and the explanations.

    Examples
    --------
    >>> import pandas as pd
    >>> from mixdist import gower_matrix
    >>> X = pd.DataFrame({"age": [20, 40, 60], "plan": ["a", "a", "b"]})
    >>> D = gower_matrix(X)
    >>> D.shape
    (3, 3)
    >>> bool(D[0, 1] < D[0, 2])
    True
    """
    metric = MixedMetric(weights=weights, **kwargs).fit(X)
    return metric.pairwise(X, Y)
