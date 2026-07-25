import numpy as np
import pandas as pd
import pytest

from mixdist import MixedMetric, make_mixed_blobs
from mixdist.stats import gini_impurity, gini_mean_difference


def test_gini_mean_difference_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    brute = np.abs(x[:, None] - x[None, :]).mean()
    assert gini_mean_difference(x) == pytest.approx(brute)


def test_gini_mean_difference_ignores_nan():
    assert gini_mean_difference(np.array([0.0, 1.0, np.nan])) == pytest.approx(0.5)


def test_gini_impurity_matches_brute_force():
    codes = np.array([0, 0, 1, 2, 2, 2])
    brute = (codes[:, None] != codes[None, :]).mean()
    assert gini_impurity(codes) == pytest.approx(brute)


def test_gini_impurity_ignores_missing_codes():
    assert gini_impurity(np.array([0, 1, -1])) == pytest.approx(0.5)


def test_weights_sum_to_column_count():
    X, _ = make_mixed_blobs(n_samples=100, random_state=0)
    for scheme in ("equal", "balanced", "type_balanced"):
        metric = MixedMetric(weights=scheme).fit(X)
        assert metric.weights_.sum() == pytest.approx(len(metric.schema_.active))


def test_balanced_equalises_the_share_of_distance():
    X, _ = make_mixed_blobs(n_samples=400, random_state=0)
    report = MixedMetric(weights="balanced").fit(X).column_report()
    assert report["share"].std() < 1e-9


def test_equal_weights_let_high_cardinality_dominate():
    X, _ = make_mixed_blobs(n_samples=400, random_state=0)
    report = MixedMetric(weights="equal").fit(X).column_report()
    noise = report.loc[["noise_0", "noise_1"], "share"].mean()
    numeric = report.loc[["num_0", "num_1", "num_2"], "share"].mean()
    assert noise > 4 * numeric  # per column, 30 levels of pure noise beat real signal


def test_type_balanced_splits_the_two_blocks_evenly():
    X, _ = make_mixed_blobs(n_samples=200, random_state=0)
    metric = MixedMetric(weights="type_balanced").fit(X)
    assert metric.w_num_.sum() == pytest.approx(metric.w_cat_.sum())


def test_mapping_weights_are_partial():
    X, _ = make_mixed_blobs(n_samples=60, random_state=0)
    metric = MixedMetric(weights={"num_0": 10.0}).fit(X)
    w = metric.weights_
    assert w["num_0"] > w["num_1"]
    assert w["num_1"] == pytest.approx(w["cat_0"])


def test_sequence_weights_follow_schema_order():
    X, _ = make_mixed_blobs(n_samples=60, random_state=0)
    schema_names = MixedMetric().fit(X).schema_.names
    values = np.arange(1, len(schema_names) + 1, dtype=float)
    metric = MixedMetric(weights=values).fit(X)
    assert list(metric.weights_.index) == schema_names
    assert metric.weights_.iloc[-1] > metric.weights_.iloc[0]


def test_unknown_scheme_raises():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(ValueError, match="Unknown weighting scheme"):
        MixedMetric(weights="magic").fit(X)


def test_unknown_column_in_mapping_raises():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(KeyError, match="unknown column"):
        MixedMetric(weights={"nope": 1.0}).fit(X)


def test_wrong_length_sequence_raises():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(ValueError, match="Expected 2 weights"):
        MixedMetric(weights=[1.0]).fit(X)


def test_negative_weight_raises():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(ValueError, match="negative"):
        MixedMetric(weights=[-1.0, 1.0]).fit(X)


def test_all_zero_weights_raise():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    with pytest.raises(ValueError, match="All weights are zero"):
        MixedMetric(weights=[0.0, 0.0]).fit(X)
