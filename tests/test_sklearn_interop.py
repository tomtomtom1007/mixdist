"""mixdist has no scikit-learn dependency, but must cooperate when it is installed."""

import numpy as np
import pytest

from mixdist import KPrototypes, MixedMetric, make_mixed_blobs

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return make_mixed_blobs(n_samples=300, random_state=0)


def test_clone_roundtrips_parameters():
    metric = MixedMetric(weights="balanced", n_bins=64, n_jobs=2)
    assert clone(metric).get_params() == metric.get_params()
    assert clone(KPrototypes(4, gamma=0.3)).get_params()["gamma"] == 0.3


def test_precomputed_distances_feed_agglomerative(data):
    X, y = data
    D = MixedMetric(weights="balanced").fit(X).pairwise(X)
    labels = AgglomerativeClustering(
        n_clusters=3, metric="precomputed", linkage="complete"
    ).fit_predict(D)
    assert adjusted_rand_score(y, labels) > 0.4


def test_precomputed_distances_feed_dbscan(data):
    X, _ = data
    D = MixedMetric().fit(X).pairwise(X)
    labels = DBSCAN(eps=float(np.quantile(D, 0.02)), min_samples=5, metric="precomputed")
    assert labels.fit_predict(D).shape == (len(X),)


def test_embedding_feeds_kmeans(data):
    X, y = data
    Z = MixedMetric(weights="balanced").fit_transform(X)
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Z)
    assert adjusted_rand_score(y, labels) > 0.6


def test_embedding_silhouette_agrees_with_precomputed_gower(data):
    X, _ = data
    metric = MixedMetric(n_bins=1024).fit(X)
    labels = KPrototypes(3, random_state=0, n_init=3).fit_predict(X)

    from_precomputed = silhouette_score(metric.pairwise(X), labels, metric="precomputed")
    Z = metric.transform(X, dtype=np.float64)
    # Euclidean distance in the embedding is sqrt(Gower), a monotone transform.
    from_embedding = silhouette_score(Z**1.0, labels, metric="euclidean")
    assert np.sign(from_precomputed) == np.sign(from_embedding)


def test_metric_works_as_a_pipeline_transformer(data):
    X, y = data
    pipe = Pipeline(
        [
            ("gower", MixedMetric(weights="balanced", n_bins=64)),
            ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=0)),
        ]
    )
    labels = pipe.fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.5
