"""Cross-validation against the reference `gower` package.

`mixdist` reimplements Gower's coefficient rather than wrapping anything, so
agreement with an independent implementation is the strongest correctness
evidence available.  These tests are skipped when `gower` is not installed
(`pip install gower`), and they pin `weights="equal"`, which is Gower (1971) as
published.
"""

import numpy as np
import pandas as pd
import pytest

from mixdist import MixedMetric, make_mixed_blobs

gower = pytest.importorskip("gower")

# The reference package predates pandas' StringDtype and cannot consume it.
OBJECT_DTYPE_REQUIRED = "the reference package needs object-dtype categoricals"


def as_object_dtype(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in out.columns:
        if out[name].dtype != float:
            out[name] = out[name].astype(object)
    return out


def reference_matrix(frame: pd.DataFrame) -> np.ndarray:
    try:
        return gower.gower_matrix(as_object_dtype(frame))
    except (TypeError, ValueError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{OBJECT_DTYPE_REQUIRED}: {exc}")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_implementation(seed):
    X, _ = make_mixed_blobs(n_samples=200, random_state=seed)
    ours = MixedMetric(weights="equal").fit(X).pairwise(X)
    theirs = reference_matrix(X)
    # The reference computes in float32, so ~1e-7 is the achievable agreement.
    assert np.abs(ours - theirs).max() < 1e-6


def test_matches_reference_on_numeric_only_data():
    X, _ = make_mixed_blobs(
        n_samples=150, n_nominal=0, n_noise_nominal=0, random_state=0
    )
    ours = MixedMetric(weights="equal").fit(X).pairwise(X)
    assert np.abs(ours - reference_matrix(X)).max() < 1e-6


def test_matches_reference_on_categorical_only_data():
    X, _ = make_mixed_blobs(
        n_samples=150, n_numeric=0, n_nominal=3, n_noise_nominal=1, random_state=0
    )
    ours = MixedMetric(weights="equal").fit(X).pairwise(X)
    assert np.abs(ours - reference_matrix(X)).max() < 1e-6


def test_matches_reference_on_a_rectangular_pair():
    X, _ = make_mixed_blobs(n_samples=120, random_state=4)
    query, reference = X.iloc[:20], X.iloc[20:]
    ours = MixedMetric(weights="equal").fit(X).pairwise(query, reference)
    try:
        theirs = gower.gower_matrix(
            as_object_dtype(query), as_object_dtype(reference)
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover
        pytest.skip(f"{OBJECT_DTYPE_REQUIRED}: {exc}")
    assert np.abs(ours - theirs).max() < 1e-6
