import numpy as np
import pandas as pd
import pytest

from mixdist import KAMILA, KPrototypes, MixedMetric, make_mixed_blobs

sklearn_metrics = pytest.importorskip("sklearn.metrics")
adjusted_rand_score = sklearn_metrics.adjusted_rand_score

ESTIMATORS = [KPrototypes, KAMILA]


@pytest.fixture(scope="module")
def data():
    return make_mixed_blobs(n_samples=400, random_state=0)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_recovers_known_clusters(estimator, data):
    X, y = data
    labels = estimator(n_clusters=3, random_state=0, n_init=5).fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.6


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_produces_exactly_n_clusters(estimator, data):
    X, _ = data
    model = estimator(n_clusters=5, random_state=0, n_init=3).fit(X)
    assert len(np.unique(model.labels_)) == 5
    assert model.labels_.shape == (len(X),)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_predict_reproduces_training_labels(estimator, data):
    X, _ = data
    model = estimator(n_clusters=3, random_state=0, n_init=5).fit(X)
    assert (model.predict(X) == model.labels_).mean() > 0.95


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_is_deterministic_given_a_seed(estimator, data):
    X, _ = data
    a = estimator(n_clusters=3, random_state=7, n_init=2).fit_predict(X)
    b = estimator(n_clusters=3, random_state=7, n_init=2).fit_predict(X)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_handles_missing_values(estimator):
    X, y = make_mixed_blobs(n_samples=250, missing_rate=0.1, random_state=1)
    labels = estimator(n_clusters=3, random_state=0, n_init=3).fit_predict(X)
    assert len(np.unique(labels)) == 3
    assert adjusted_rand_score(y, labels) > 0.3


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_numeric_only_table(estimator):
    X, y = make_mixed_blobs(n_samples=200, n_nominal=0, n_noise_nominal=0, random_state=2)
    labels = estimator(n_clusters=3, random_state=0, n_init=3).fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.6


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_categorical_only_table(estimator):
    X, y = make_mixed_blobs(
        n_samples=300, n_numeric=0, n_nominal=4, n_noise_nominal=0, purity=0.95,
        random_state=3,
    )
    labels = estimator(n_clusters=3, random_state=0, n_init=5).fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.4


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_accepts_a_prefitted_metric(estimator, data):
    X, _ = data
    metric = MixedMetric(weights="balanced").fit(X)
    model = estimator(n_clusters=3, metric=metric, random_state=0, n_init=2).fit(X)
    assert model.metric_ is metric


def test_kprototypes_auto_gamma_is_positive(data):
    X, _ = data
    assert KPrototypes(3, random_state=0, n_init=1).fit(X).gamma_ > 0


def test_kprototypes_modha_spangler_picks_its_own_gamma(data):
    X, y = data
    model = KPrototypes(3, gamma="modha-spangler", random_state=0, n_init=2).fit(X)
    assert model.gamma_ > 0
    assert adjusted_rand_score(y, model.labels_) > 0.6


def test_kprototypes_respects_an_explicit_gamma(data):
    X, _ = data
    assert KPrototypes(3, gamma=0.25, random_state=0, n_init=1).fit(X).gamma_ == 0.25


def test_kprototypes_centres_are_in_original_units(data):
    X, _ = data
    centres = KPrototypes(3, random_state=0, n_init=2).fit(X).cluster_centers_
    assert list(centres.columns) == list(X.columns)
    assert centres["num_0"].between(X["num_0"].min(), X["num_0"].max()).all()
    assert set(centres["cat_0"]) <= set(X["cat_0"])


def test_kamila_centres_and_level_probabilities(data):
    X, _ = data
    model = KAMILA(3, random_state=0, n_init=3).fit(X)
    centres = model.cluster_centers_
    assert list(centres.columns) == list(X.columns)

    probs = model.level_probabilities_["cat_0"]
    assert probs.shape == (3, X["cat_0"].nunique())
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_kamila_needs_no_gamma():
    assert "gamma" not in KAMILA._param_names()


def test_more_clusters_than_rows_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    with pytest.raises(ValueError, match="n_clusters must lie"):
        KPrototypes(n_clusters=10, random_state=0, n_init=1).fit(X)


def test_duplicate_rows_do_not_break_seeding():
    X = pd.DataFrame({"a": [1.0] * 20, "b": ["x"] * 20})
    X.loc[0, "a"] = 2.0
    model = KPrototypes(n_clusters=3, random_state=0, n_init=1).fit(X)
    assert len(np.unique(model.labels_)) == 3
