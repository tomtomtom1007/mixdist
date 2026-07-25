import numpy as np
import pandas as pd
import pytest

from mixdist.schema import Schema, infer_kinds


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "age": [20.0, 35.0, 50.0, 65.0],
            "spend": [100.0, 250.0, 90.0, 700.0],
            "plan": ["free", "pro", "free", "team"],
            "flag": [True, False, True, True],
            "tier": pd.Categorical(["low", "high", "mid", "low"],
                                   categories=["low", "mid", "high"], ordered=True),
            "n_seats": [1, 2, 1, 3],
        }
    )


def test_infer_kinds(frame):
    kinds = infer_kinds(frame)
    assert kinds["age"] == "numeric"
    assert kinds["plan"] == "nominal"
    assert kinds["flag"] == "nominal"
    assert kinds["tier"] == "ordinal"
    assert kinds["n_seats"] == "nominal"  # low-cardinality integer


def test_infer_kinds_respects_max_int_levels(frame):
    assert infer_kinds(frame, max_int_levels=1)["n_seats"] == "numeric"


def test_encode_shapes_and_ranges(frame):
    schema = Schema.fit(frame)
    num, cat = schema.encode(frame)
    assert num.shape == (4, schema.n_numeric)
    assert cat.shape == (4, schema.n_nominal)
    assert np.nanmin(num) >= 0.0 and np.nanmax(num) <= 1.0
    assert cat.min() >= 0


def test_ordinal_becomes_numeric_in_rank_order(frame):
    schema = Schema.fit(frame)
    tier = next(c for c in schema.numeric if c.name == "tier")
    num, _ = schema.encode(frame)
    values = num[:, tier.slot]
    # low < mid < high, and the categorical order wins over alphabetical order.
    assert values[0] == 0.0 and values[1] == 1.0 and values[2] == 0.5


def test_explicit_kinds_override_inference(frame):
    schema = Schema.fit(frame, kinds={"n_seats": "numeric"})
    assert "n_seats" in [c.name for c in schema.numeric]


def test_categorical_and_ordinal_shorthands(frame):
    schema = Schema.fit(frame, categorical=["age"])
    assert "age" in [c.name for c in schema.nominal]


def test_constant_column_is_dropped_with_warning():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "same": ["x", "x", "x"]})
    with pytest.warns(UserWarning, match="constant"):
        schema = Schema.fit(frame)
    assert [c.name for c in schema.active] == ["a"]
    assert [c.name for c in schema.dropped] == ["same"]


def test_constant_column_can_be_kept():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "same": ["x", "x", "x"]})
    schema = Schema.fit(frame, drop_constant=False)
    assert len(schema.active) == 2


def test_all_constant_raises():
    frame = pd.DataFrame({"a": [1.0, 1.0], "b": ["x", "x"]})
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="No usable columns"):
            Schema.fit(frame)


def test_missing_values_survive_encoding():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    schema = Schema.fit(frame)
    num, cat = schema.encode(frame)
    assert np.isnan(num[1, 0])
    assert cat[2, 0] == -1


def test_unseen_category_encodes_as_missing():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    schema = Schema.fit(train)
    _, cat = schema.encode(pd.DataFrame({"a": [1.5], "b": ["z"]}))
    assert cat[0, 0] == -1


def test_out_of_range_numeric_is_clipped():
    schema = Schema.fit(pd.DataFrame({"a": [0.0, 10.0]}))
    num, _ = schema.encode(pd.DataFrame({"a": [-5.0, 15.0]}))
    assert num[0, 0] == 0.0 and num[1, 0] == 1.0


def test_robust_range_resists_outliers():
    values = np.concatenate([np.linspace(0, 1, 99), [1000.0]])
    frame = pd.DataFrame({"a": values})
    minmax, _ = Schema.fit(frame, numeric_range="minmax").encode(frame)
    robust, _ = Schema.fit(frame, numeric_range="robust", clip_quantile=0.05).encode(frame)
    # Under minmax the outlier squashes everything else into a sliver.
    assert np.ptp(minmax[:99]) < 0.01
    assert np.ptp(robust[:99]) > 0.9


def test_missing_required_column_raises():
    schema = Schema.fit(pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}))
    with pytest.raises(ValueError, match="Missing column"):
        schema.encode(pd.DataFrame({"a": [1.0]}))


def test_numpy_input_is_accepted():
    schema = Schema.fit(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]]))
    assert schema.n_numeric == 2


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown column kind"):
        Schema.fit(pd.DataFrame({"a": [1.0, 2.0]}), kinds={"a": "banana"})


def test_decode_scaled_roundtrip(frame):
    schema = Schema.fit(frame)
    age = next(c for c in schema.numeric if c.name == "age")
    assert age.decode_scaled(np.array([0.0, 1.0])).tolist() == [20.0, 65.0]
    tier = next(c for c in schema.numeric if c.name == "tier")
    assert tier.decode_scaled(np.array([1.0]))[0] == "high"
