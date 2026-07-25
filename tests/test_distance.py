import numpy as np
import pandas as pd
import pytest

from mixdist import MixedMetric, gower_matrix, make_mixed_blobs


@pytest.fixture
def toy():
    return pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "plan": ["a", "a", "b"],
        }
    )


def test_matches_hand_computed_gower(toy):
    D = gower_matrix(toy)  # equal weights, range(age) = 20
    # rows 0,1: |20-30|/20 = 0.5 numeric, 0 categorical -> (0.5 + 0) / 2
    assert D[0, 1] == pytest.approx(0.25)
    # rows 0,2: |20-40|/20 = 1.0 numeric, 1 categorical -> (1 + 1) / 2
    assert D[0, 2] == pytest.approx(1.0)
    assert D[1, 2] == pytest.approx((0.5 + 1.0) / 2)


def test_symmetry_zero_diagonal_and_unit_range(toy):
    D = gower_matrix(toy)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert D.min() >= 0.0 and D.max() <= 1.0


def test_numeric_only_reduces_to_normalised_l1():
    X = pd.DataFrame({"a": [0.0, 1.0, 4.0], "b": [0.0, 2.0, 4.0]})
    D = gower_matrix(X)
    scaled = X.to_numpy() / 4.0
    expected = np.abs(scaled[:, None, :] - scaled[None, :, :]).mean(axis=2)
    assert np.allclose(D, expected)


def test_categorical_only_reduces_to_hamming():
    X = pd.DataFrame({"a": ["x", "x", "y"], "b": ["p", "q", "q"]})
    D = gower_matrix(X)
    assert D[0, 1] == pytest.approx(0.5)
    assert D[0, 2] == pytest.approx(1.0)


def test_blocking_does_not_change_results():
    X, _ = make_mixed_blobs(n_samples=120, random_state=0)
    big = MixedMetric().fit(X)
    small = MixedMetric(block_elements=97).fit(X)
    assert np.allclose(big.pairwise(X), small.pairwise(X))


def test_threads_do_not_change_results():
    X, _ = make_mixed_blobs(n_samples=120, random_state=0)
    serial = MixedMetric(block_elements=97).fit(X)
    threaded = MixedMetric(block_elements=97, n_jobs=4).fit(X)
    assert np.allclose(serial.pairwise(X), threaded.pairwise(X))


def test_iter_pairwise_reassembles_the_matrix():
    X, _ = make_mixed_blobs(n_samples=90, random_state=0)
    metric = MixedMetric(block_elements=200).fit(X)
    full = metric.pairwise(X)
    out = np.empty_like(full)
    for start, stop, block in metric.iter_pairwise(X):
        out[start:stop] = block
    assert np.allclose(out, full)


def test_missing_values_drop_out_of_the_pair_denominator():
    X = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": ["x", "y", "x"]})
    metric = MixedMetric(weights="equal").fit(X)
    D = metric.pairwise(X)
    # Row 2 has no numeric value, so only 'b' contributes and it matches row 0.
    assert D[0, 2] == pytest.approx(0.0)
    assert D[1, 2] == pytest.approx(1.0)


def test_all_missing_pair_is_nan():
    X = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": ["x", "y", None]})
    metric = MixedMetric(weights="equal").fit(X)
    D = metric.pairwise(X)
    assert np.isnan(D[2, 0])


def test_kneighbors_is_exact():
    X, _ = make_mixed_blobs(n_samples=150, random_state=3)
    metric = MixedMetric(block_elements=500).fit(X)
    D = metric.pairwise(X)
    dist, idx = metric.kneighbors(n_neighbors=7)
    reference = np.argsort(D + np.eye(len(D)) * 1e9, axis=1)[:, :7]
    assert np.array_equal(idx, reference)
    assert np.allclose(dist, np.take_along_axis(D, reference, axis=1))


def test_kneighbors_against_separate_reference_set():
    X, _ = make_mixed_blobs(n_samples=80, random_state=4)
    metric = MixedMetric().fit(X)
    query, reference = X.iloc[:10], X.iloc[10:]
    D = metric.pairwise(query, reference)
    dist, idx = metric.kneighbors(query, reference, n_neighbors=3)
    assert np.allclose(dist, np.take_along_axis(D, idx, axis=1))
    assert np.allclose(dist, np.sort(D, axis=1)[:, :3])


def test_kneighbors_rejects_impossible_k(toy):
    metric = MixedMetric().fit(toy)
    with pytest.raises(ValueError, match="k must lie"):
        metric.kneighbors(n_neighbors=99)


def test_rectangular_pairwise():
    X, _ = make_mixed_blobs(n_samples=40, random_state=5)
    metric = MixedMetric().fit(X)
    D = metric.pairwise(X.iloc[:7], X.iloc[7:20])
    assert D.shape == (7, 13)


def test_zero_weight_column_is_ignored(toy):
    metric = MixedMetric(weights={"plan": 0.0}).fit(toy)
    D = metric.pairwise(toy)
    assert D[0, 2] == pytest.approx(1.0)  # numeric term only
