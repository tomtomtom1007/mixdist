"""Variable weighting schemes.

Gower's coefficient is a weighted mean of per-column dissimilarities,

.. math::
   d(i, j) = \\frac{\\sum_k w_k \\delta_{ijk} d_k(i, j)}{\\sum_k w_k \\delta_{ijk}},

and the classical choice :math:`w_k = 1` is the source of most complaints about
mixed-type distances: it equalises *nominal weight*, not *actual contribution*.
The schemes here make that choice explicit and reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .schema import Schema
from .stats import expected_dissimilarity

SCHEMES = ("equal", "balanced", "type_balanced")


def resolve_weights(
    scheme,
    schema: Schema,
    num: np.ndarray,
    cat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(w_num, w_cat)``, normalised so the weights sum to the column count.

    Parameters
    ----------
    scheme
        One of:

        ``"equal"``
            :math:`w_k = 1`.  Gower (1971) as published.  Reproducible and
            standard, but lets high-cardinality nominal columns dominate.
        ``"balanced"``
            :math:`w_k \\propto 1 / E[d_k]`, so every column contributes the
            same *expected* distance.  Cardinality and marginal skew stop
            deciding importance.  Recommended default for exploratory work.
        ``"type_balanced"``
            The numeric block and the nominal block each receive half the total
            weight, split equally inside a block.  The classical fix, useful
            when you want the two *kinds* of variable balanced but the columns
            within a kind left alone.

        A mapping ``{column: weight}`` or a sequence aligned with
        ``schema.names`` sets weights directly.  A mapping may be partial;
        unlisted columns default to ``1``.
    """
    n_num, n_cat = schema.n_numeric, schema.n_nominal
    total = n_num + n_cat

    if isinstance(scheme, str):
        w_num, w_cat = _named_scheme(scheme, schema, num, cat)
    elif isinstance(scheme, Mapping):
        w_num, w_cat = _from_mapping(scheme, schema)
    elif isinstance(scheme, Sequence) or isinstance(scheme, np.ndarray):
        w_num, w_cat = _from_sequence(scheme, schema)
    else:
        raise TypeError(
            f"weights must be one of {SCHEMES}, a mapping, or a sequence; got {type(scheme)!r}."
        )

    _validate(w_num, w_cat, schema)
    scale = float(w_num.sum() + w_cat.sum())
    if scale <= 0:
        raise ValueError("All weights are zero; at least one column must be weighted.")
    factor = total / scale
    return w_num * factor, w_cat * factor


# --------------------------------------------------------------------------- #
def _named_scheme(scheme: str, schema: Schema, num, cat):
    n_num, n_cat = schema.n_numeric, schema.n_nominal
    if scheme == "equal":
        return np.ones(n_num), np.ones(n_cat)

    if scheme == "balanced":
        d_num, d_cat = expected_dissimilarity(num, cat)
        return _reciprocal(d_num), _reciprocal(d_cat)

    if scheme == "type_balanced":
        w_num = np.full(n_num, 0.5 / n_num) if n_num else np.zeros(0)
        w_cat = np.full(n_cat, 0.5 / n_cat) if n_cat else np.zeros(0)
        if not n_num:  # single-type table: fall back to equal weights
            w_cat = np.ones(n_cat)
        if not n_cat:
            w_num = np.ones(n_num)
        return w_num, w_cat

    raise ValueError(f"Unknown weighting scheme {scheme!r}; expected one of {SCHEMES}.")


def _reciprocal(d: np.ndarray) -> np.ndarray:
    """``1 / d`` with zero-dispersion columns pinned to zero weight."""
    out = np.zeros_like(d, dtype=np.float64)
    positive = d > 0
    out[positive] = 1.0 / d[positive]
    return out


def _from_mapping(mapping: Mapping, schema: Schema):
    known = {c.name for c in schema.active}
    unknown = set(mapping) - known - {c.name for c in schema.dropped}
    if unknown:
        raise KeyError(f"Weights given for unknown column(s): {sorted(map(str, unknown))}.")
    w_num = np.array([float(mapping.get(c.name, 1.0)) for c in schema.numeric])
    w_cat = np.array([float(mapping.get(c.name, 1.0)) for c in schema.nominal])
    return w_num, w_cat


def _from_sequence(seq, schema: Schema):
    values = np.asarray(seq, dtype=np.float64).ravel()
    names = schema.names
    if values.size != len(names):
        raise ValueError(
            f"Expected {len(names)} weights (one per active column, ordered as "
            f"{names}), got {values.size}."
        )
    return values[: schema.n_numeric].copy(), values[schema.n_numeric :].copy()


def _validate(w_num: np.ndarray, w_cat: np.ndarray, schema: Schema) -> None:
    blocks = (
        (w_num, schema.n_numeric, "numeric"),
        (w_cat, schema.n_nominal, "nominal"),
    )
    for block, expected, label in blocks:
        if block.shape != (expected,):
            raise ValueError(f"{label} weights have shape {block.shape}, expected ({expected},).")
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{label} weights contain non-finite values.")
        if np.any(block < 0):
            raise ValueError(f"{label} weights contain negative values.")
