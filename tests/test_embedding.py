import numpy as np
import pandas as pd
import pytest

from mixdist import MixedMetric, make_mixed_blobs


def squared_distances(Z):
    return ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)


@pytest.mark.parametrize("weights", ["equal", "balanced", "type_balanced"])
def test_squared_euclidean_equals_gower(weights):
    X, _ = make_mixed_blobs(n_samples=120, random_state=0)
    metric = MixedMetric(weights=weights, n_bins=1024).fit(X)
    D = metric.pairwise(X)
    Z = metric.transform(X, dtype=np.float64)
    # Quantisation error is bounded by sum_k w_k / (W * n_bins) over numeric columns.
    bound = metric.w_num_.sum() / (metric.weights_.sum() * 1024)
    assert np.abs(squared_distances(Z) - D).max() <= bound + 1e-12


def test_categorical_only_embedding_is_exact():
    X = pd.DataFrame({"a": ["x", "y", "z", "x"], "b": ["p", "p", "q", "q"]})
    metric = MixedMetric(weights="equal").fit(X)
    Z = metric.transform(X, dtype=np.float64)
    assert np.allclose(squared_distances(Z), metric.pairwise(X))


def test_quantisation_error_shrinks_with_more_bins():
    X, _ = make_mixed_blobs(n_samples=60, n_noise_nominal=0, random_state=1)
    D = MixedMetric(n_bins=16).fit(X).pairwise(X)
    errors = []
    for n_bins in (16, 256):
        metric = MixedMetric(n_bins=n_bins).fit(X)
        Z = metric.transform(X, dtype=np.float64)
        errors.append(np.abs(squared_distances(Z) - D).max())
    assert errors[1] < errors[0] / 4


def test_neighbour_ranking_is_preserved_exactly():
    X, _ = make_mixed_blobs(n_samples=100, random_state=2)
    metric = MixedMetric(n_bins=2048).fit(X)
    _, gower_idx = metric.kneighbors(n_neighbors=5)

    Z = metric.transform(X, dtype=np.float64)
    E = squared_distances(Z) + np.eye(len(X)) * 1e9
    embedded_idx = np.argsort(E, axis=1)[:, :5]
    assert (gower_idx == embedded_idx).mean() > 0.99


def test_random_projection_approximately_preserves_distances():
    X, _ = make_mixed_blobs(n_samples=200, random_state=0)
    metric = MixedMetric().fit(X)
    D = metric.pairwise(X)
    Z = metric.transform(X, n_components=256, random_state=0, dtype=np.float64)
    assert Z.shape[1] == 256
    E = squared_distances(Z)
    off = ~np.eye(len(X), dtype=bool)
    assert np.corrcoef(D[off], E[off])[0, 1] > 0.95


def test_projection_is_skipped_when_larger_than_exact_dimension():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": ["x", "y"]})
    metric = MixedMetric(n_bins=4).fit(X)
    emb = metric.embedding(n_components=10_000)
    assert emb.is_exact
    assert emb.n_features_out_ == emb.exact_n_features_


def test_feature_names_describe_the_map():
    X = pd.DataFrame({"a": [0.0, 1.0], "b": ["x", "y"]})
    names = MixedMetric(n_bins=3).fit(X).embedding().feature_names_out()
    assert names[:3] == ["a>bin0", "a>bin1", "a>bin2"]
    assert names[3:] == ["b=level0", "b=level1"]


def test_missing_values_warn_when_imputed():
    X = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": ["x", "y", "x"]})
    metric = MixedMetric().fit(X)
    with pytest.warns(UserWarning, match="Imputing"):
        metric.transform(X)


def test_missing_values_can_be_refused():
    X = pd.DataFrame({"a": [0.0, 1.0, np.nan], "b": ["x", "y", "x"]})
    metric = MixedMetric().fit(X)
    with pytest.raises(ValueError, match="Cannot embed data with missing values"):
        metric.transform(X, on_missing="error")


def test_out_of_sample_rows_embed_consistently():
    X, _ = make_mixed_blobs(n_samples=80, random_state=6)
    metric = MixedMetric(n_bins=1024).fit(X)
    train, new = X.iloc[:60], X.iloc[60:]
    Z_train = metric.transform(train, dtype=np.float64)
    Z_new = metric.transform(new, dtype=np.float64)
    cross = ((Z_new[:, None, :] - Z_train[None, :, :]) ** 2).sum(-1)
    bound = metric.w_num_.sum() / (metric.weights_.sum() * 1024)
    assert np.abs(cross - metric.pairwise(new, train)).max() <= bound + 1e-12
