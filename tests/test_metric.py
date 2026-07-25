import numpy as np
import pandas as pd
import pytest

from mixdist import MixedMetric, NotFittedError, gower_matrix, make_mixed_blobs


@pytest.fixture
def data():
    return make_mixed_blobs(n_samples=150, random_state=0)


def test_methods_require_fit():
    metric = MixedMetric()
    with pytest.raises(NotFittedError):
        metric.pairwise(pd.DataFrame({"a": [1.0, 2.0]}))


def test_column_report_shares_sum_to_one(data):
    X, _ = data
    report = MixedMetric().fit(X).column_report()
    assert report["share"].sum() == pytest.approx(1.0)
    assert list(report.index) == list(report.sort_values("share", ascending=False).index)


def test_explain_decomposition_is_exact(data):
    X, _ = data
    metric = MixedMetric().fit(X)
    D = metric.pairwise(X)
    parts = metric.explain(X, 4, 11)
    assert parts.sum() == pytest.approx(D[4, 11])
    assert set(parts.index) == set(metric.schema_.names)


def test_explain_pairs_is_row_aligned(data):
    X, _ = data
    metric = MixedMetric().fit(X)
    A, B = X.iloc[:20], X.iloc[20:40]
    parts = metric.explain_pairs(A, B)
    expected = np.diag(metric.pairwise(A, B))
    assert np.allclose(parts.sum(axis=1), expected)


def test_explain_pairs_rejects_misaligned_input(data):
    X, _ = data
    metric = MixedMetric().fit(X)
    with pytest.raises(ValueError, match="row-aligned"):
        metric.explain_pairs(X.iloc[:5], X.iloc[:6])


def test_explain_clusters_names_the_informative_columns(data):
    X, y = data
    metric = MixedMetric(weights="balanced").fit(X)
    table = metric.explain_clusters(X, y)
    assert table.shape == (3, len(metric.schema_.names))
    # Noise columns are just as scattered inside a cluster as outside, so their
    # dispersion reduction is ~0 even though they sit far from the global mode.
    noise = np.abs(table[["noise_0", "noise_1"]].to_numpy()).max()
    assert noise < 0.005
    assert table[["cat_0", "cat_1"]].to_numpy().min() > 10 * noise


def test_explain_clusters_checks_label_length(data):
    X, _ = data
    metric = MixedMetric().fit(X)
    with pytest.raises(ValueError, match="one entry per row"):
        metric.explain_clusters(X, np.zeros(3))


def test_fit_caches_training_data(data):
    X, _ = data
    metric = MixedMetric().fit(X)
    assert np.allclose(metric.pairwise(), metric.pairwise(X))


def test_get_set_params_roundtrip():
    metric = MixedMetric(weights="equal", n_bins=64)
    params = metric.get_params()
    assert params["weights"] == "equal" and params["n_bins"] == 64
    metric.set_params(n_bins=32)
    assert metric.n_bins == 32
    with pytest.raises(ValueError, match="Invalid parameter"):
        metric.set_params(nope=1)


def test_repr_shows_non_default_params_only():
    assert repr(MixedMetric()) == "MixedMetric()"
    assert "n_bins=64" in repr(MixedMetric(n_bins=64))


def test_gower_matrix_defaults_to_published_coefficient():
    X = pd.DataFrame({"age": [20.0, 40.0], "plan": ["a", "b"]})
    assert gower_matrix(X)[0, 1] == pytest.approx(1.0)


def test_gower_matrix_accepts_a_second_table():
    X, _ = make_mixed_blobs(n_samples=30, random_state=0)
    assert gower_matrix(X.iloc[:5], X.iloc[5:12]).shape == (5, 7)


def test_fit_transform_matches_fit_then_transform(data):
    X, _ = data
    a = MixedMetric(n_bins=32).fit_transform(X)
    b = MixedMetric(n_bins=32).fit(X).transform(X)
    assert np.allclose(a, b)
