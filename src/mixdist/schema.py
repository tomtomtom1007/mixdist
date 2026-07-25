"""Column typing and encoding for mixed-type tables.

The schema is the single place where "what kind of variable is this?" is
decided.  Everything downstream (distances, embeddings, clusterers) consumes
the two dense arrays produced by :meth:`Schema.encode`:

``num``
    ``(n, P)`` float64, each column scaled to ``[0, 1]``; ``nan`` marks missing.
``cat``
    ``(n, Q)`` int32 category codes; ``-1`` marks missing.

Ordinal columns are rank-transformed and then handled as numeric, which is the
standard treatment in Gower's coefficient.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Kind = Literal["numeric", "nominal", "ordinal"]
KINDS = ("numeric", "nominal", "ordinal")

#: Integer columns with at most this many distinct values are inferred as
#: nominal rather than numeric (e.g. a 0/1 flag or a 5-level survey answer).
DEFAULT_MAX_INT_LEVELS = 10


@dataclass
class Column:
    """A single typed column together with the state learned from the data."""

    name: str
    kind: Kind
    #: Position within ``num`` or ``cat`` after encoding.
    slot: int = -1
    #: Numeric/ordinal: scaling bounds. ``hi == lo`` means the column is constant.
    lo: float = np.nan
    hi: float = np.nan
    #: Nominal/ordinal: observed category values, in code order.
    categories: np.ndarray | None = None
    #: True when the column takes a single value (it can carry no distance).
    constant: bool = False

    @property
    def is_numeric(self) -> bool:
        return self.kind in ("numeric", "ordinal")

    @property
    def n_levels(self) -> int:
        return 0 if self.categories is None else int(len(self.categories))

    def decode_scaled(self, u: np.ndarray) -> np.ndarray:
        """Map scaled values in ``[0, 1]`` back to original units.

        Ordinal columns are snapped to the nearest level label so that a
        reported centroid is a value the variable can actually take.
        """
        values = self.lo + np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0) * (self.hi - self.lo)
        if self.kind == "ordinal" and self.categories is not None:
            idx = np.clip(np.rint(values).astype(int), 0, len(self.categories) - 1)
            return np.asarray(self.categories)[idx]
        return values


def infer_kinds(
    frame: pd.DataFrame,
    *,
    max_int_levels: int = DEFAULT_MAX_INT_LEVELS,
) -> dict[str, Kind]:
    """Guess a :class:`Kind` for every column of ``frame``.

    Rules, in order:

    * ``pandas.CategoricalDtype`` with ``ordered=True`` -> ``ordinal``
    * any other categorical / object / string / bool dtype -> ``nominal``
    * integer dtype with ``<= max_int_levels`` distinct values -> ``nominal``
    * remaining numeric dtypes -> ``numeric``

    The guess is a convenience, not a contract.  Pass ``kinds=`` explicitly for
    anything that matters.
    """
    kinds: dict[str, Kind] = {}
    for name in frame.columns:
        s = frame[name]
        dtype = s.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            kinds[name] = "ordinal" if dtype.ordered else "nominal"
        elif pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
            kinds[name] = "nominal"
        elif pd.api.types.is_integer_dtype(dtype) and s.nunique(dropna=True) <= max_int_levels:
            kinds[name] = "nominal"
        else:
            kinds[name] = "numeric"
    return kinds


@dataclass
class Schema:
    """Typed, fitted description of a table.

    Use :meth:`fit` to build one; :meth:`encode` to turn a frame into the dense
    arrays the rest of the library works on.
    """

    columns: list[Column] = field(default_factory=list)
    numeric_range: str = "minmax"
    drop_constant: bool = True

    # ------------------------------------------------------------------ fit --
    @classmethod
    def fit(
        cls,
        X,
        *,
        kinds: Mapping[str, Kind] | Sequence[Kind] | None = None,
        categorical: Iterable[str | int] | None = None,
        ordinal: Iterable[str | int] | None = None,
        numeric_range: str = "minmax",
        clip_quantile: float = 0.01,
        drop_constant: bool = True,
        max_int_levels: int = DEFAULT_MAX_INT_LEVELS,
    ) -> Schema:
        """Learn a schema from ``X``.

        Parameters
        ----------
        X
            ``DataFrame``, 2-D array, or anything ``pandas`` can wrap.
        kinds
            Explicit ``{column: kind}`` mapping (or a sequence aligned with the
            columns).  Columns left out fall back to :func:`infer_kinds`.
        categorical, ordinal
            Convenience shorthands; equivalent to listing those columns in
            ``kinds`` as ``"nominal"`` / ``"ordinal"``.
        numeric_range
            ``"minmax"`` reproduces Gower's original coefficient.  ``"robust"``
            uses the ``[clip_quantile, 1 - clip_quantile]`` range and clips
            beyond it, so a single outlier cannot flatten a whole variable.
        clip_quantile
            Tail probability used by ``numeric_range="robust"``.
        drop_constant
            Drop columns with a single observed value.  They contribute no
            distance but do enter Gower's denominator, which silently rescales
            every distance.
        """
        frame = as_frame(X)
        resolved = _resolve_kinds(frame, kinds, categorical, ordinal, max_int_levels)
        if numeric_range not in ("minmax", "robust"):
            raise ValueError("numeric_range must be 'minmax' or 'robust'")
        if not 0.0 <= clip_quantile < 0.5:
            raise ValueError("clip_quantile must lie in [0, 0.5)")

        columns: list[Column] = []
        n_num = n_cat = 0
        for name in frame.columns:
            col = _fit_column(
                frame[name],
                str(name),
                resolved[name],
                numeric_range=numeric_range,
                clip_quantile=clip_quantile,
            )
            if col.constant and drop_constant:
                columns.append(col)
                continue
            if col.is_numeric:
                col.slot = n_num
                n_num += 1
            else:
                col.slot = n_cat
                n_cat += 1
            columns.append(col)

        schema = cls(columns=columns, numeric_range=numeric_range, drop_constant=drop_constant)
        dropped = [c.name for c in schema.dropped]
        if dropped:
            warnings.warn(
                f"Dropping constant column(s) {dropped}; they carry no information "
                "and would rescale every distance. Pass drop_constant=False to keep them.",
                UserWarning,
                stacklevel=2,
            )
        if not schema.active:
            raise ValueError("No usable columns: every column is constant or empty.")
        return schema

    # -------------------------------------------------------------- accessors -
    @property
    def active(self) -> list[Column]:
        """Columns that participate in distance computations."""
        return [c for c in self.columns if c.slot >= 0]

    @property
    def dropped(self) -> list[Column]:
        return [c for c in self.columns if c.slot < 0]

    @property
    def numeric(self) -> list[Column]:
        return sorted((c for c in self.active if c.is_numeric), key=lambda c: c.slot)

    @property
    def nominal(self) -> list[Column]:
        return sorted((c for c in self.active if not c.is_numeric), key=lambda c: c.slot)

    @property
    def names(self) -> list[str]:
        """Active column names, numeric block first, then nominal."""
        return [c.name for c in self.numeric] + [c.name for c in self.nominal]

    @property
    def n_numeric(self) -> int:
        return len(self.numeric)

    @property
    def n_nominal(self) -> int:
        return len(self.nominal)

    # ---------------------------------------------------------------- encode --
    def encode(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Encode ``X`` into ``(num, cat)`` using the fitted state.

        Unseen categories encode to ``-1`` (treated as missing), and numeric
        values outside the fitted range are clipped into ``[0, 1]``.
        """
        frame = as_frame(X)
        missing = [c.name for c in self.active if c.name not in frame.columns]
        if missing:
            raise ValueError(f"Missing column(s) required by the schema: {missing}")

        n = len(frame)
        num = np.empty((n, self.n_numeric), dtype=np.float64)
        cat = np.empty((n, self.n_nominal), dtype=np.int32)

        for col in self.numeric:
            num[:, col.slot] = _encode_numeric(frame[col.name], col)
        for col in self.nominal:
            cat[:, col.slot] = _encode_nominal(frame[col.name], col)
        return num, cat


# ------------------------------------------------------------------ helpers --
def as_frame(X) -> pd.DataFrame:
    """Coerce ``X`` to a ``DataFrame`` without copying when avoidable."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D table, got array with shape {arr.shape}.")
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def _resolve_kinds(
    frame: pd.DataFrame,
    kinds,
    categorical,
    ordinal,
    max_int_levels: int,
) -> dict[str, Kind]:
    resolved: dict[str, Kind] = dict(infer_kinds(frame, max_int_levels=max_int_levels))

    if kinds is not None:
        if isinstance(kinds, Mapping):
            items = kinds.items()
        else:
            kinds = list(kinds)
            if len(kinds) != frame.shape[1]:
                raise ValueError(
                    f"kinds has length {len(kinds)} but the table has {frame.shape[1]} columns."
                )
            items = zip(frame.columns, kinds)
        for key, kind in items:
            resolved[_column_key(frame, key)] = _check_kind(kind)

    for group, kind in ((categorical, "nominal"), (ordinal, "ordinal")):
        for key in group or ():
            resolved[_column_key(frame, key)] = kind  # type: ignore[assignment]
    return resolved


def _column_key(frame: pd.DataFrame, key) -> str:
    if key in frame.columns:
        return key
    if isinstance(key, (int, np.integer)) and 0 <= int(key) < frame.shape[1]:
        return frame.columns[int(key)]
    raise KeyError(f"Unknown column {key!r}.")


def _check_kind(kind) -> Kind:
    if kind not in KINDS:
        raise ValueError(f"Unknown column kind {kind!r}; expected one of {KINDS}.")
    return kind  # type: ignore[return-value]


def _fit_column(
    series: pd.Series,
    name: str,
    kind: Kind,
    *,
    numeric_range: str,
    clip_quantile: float,
) -> Column:
    col = Column(name=name, kind=kind)

    if kind == "nominal":
        values = pd.Series(series.to_numpy(), dtype="object")
        cats = pd.Index(values.dropna().unique())
        col.categories = cats.to_numpy()
        col.constant = len(cats) <= 1
        return col

    if kind == "ordinal":
        cats = _ordinal_categories(series, name)
        col.categories = cats
        codes = _match_categories(series, cats).astype(np.float64)
        codes[codes < 0] = np.nan
        col.lo, col.hi = 0.0, float(max(len(cats) - 1, 0))
        col.constant = col.hi <= 0 or np.all(np.isnan(codes))
        return col

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        col.lo = col.hi = np.nan
        col.constant = True
        return col
    if numeric_range == "robust" and clip_quantile > 0:
        lo, hi = np.quantile(finite, [clip_quantile, 1.0 - clip_quantile])
        if hi <= lo:  # degenerate quantiles (very skewed / tiny sample)
            lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = float(finite.min()), float(finite.max())
    col.lo, col.hi = float(lo), float(hi)
    col.constant = not (col.hi > col.lo)
    return col


def _ordinal_categories(series: pd.Series, name: str) -> np.ndarray:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        if not dtype.ordered:
            warnings.warn(
                f"Column {name!r} is declared ordinal but its CategoricalDtype is unordered; "
                "using the dtype's category order as-is.",
                UserWarning,
                stacklevel=3,
            )
        return np.asarray(dtype.categories)
    values = series.dropna().unique()
    try:
        return np.sort(values)
    except TypeError as exc:  # pragma: no cover - mixed unorderable types
        raise TypeError(
            f"Column {name!r} is declared ordinal but its values are not sortable. "
            "Convert it to an ordered pandas Categorical first."
        ) from exc


def _match_categories(series: pd.Series, categories: np.ndarray) -> np.ndarray:
    """Map values to positions in ``categories``; ``-1`` for missing/unseen."""
    if isinstance(series.dtype, pd.CategoricalDtype) and np.array_equal(
        np.asarray(series.dtype.categories), categories
    ):
        return series.cat.codes.to_numpy().astype(np.int64)
    index = pd.Index(categories)
    values = pd.Series(series.to_numpy(), dtype="object")
    codes = index.get_indexer(values)  # -1 for unseen
    codes = np.asarray(codes, dtype=np.int64)
    codes[pd.isna(values).to_numpy()] = -1
    return codes


def _encode_numeric(series: pd.Series, col: Column) -> np.ndarray:
    if col.kind == "ordinal":
        assert col.categories is not None
        values = _match_categories(series, col.categories).astype(np.float64)
        values[values < 0] = np.nan
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values), values, np.nan)
    span = col.hi - col.lo
    if not span > 0:
        return np.where(np.isnan(values), np.nan, 0.0)
    scaled = (values - col.lo) / span
    return np.clip(scaled, 0.0, 1.0)


def _encode_nominal(series: pd.Series, col: Column) -> np.ndarray:
    assert col.categories is not None
    return _match_categories(series, col.categories).astype(np.int32)
