# mixdist

**Distances, nearest-neighbour search and clustering for mixed numeric/categorical tables.**

*[日本語 README](README.ja.md)*

Real customer tables mix `age` and `spend` with `gender` and `plan_tier`. The standard
answers — Gower's distance and k-prototypes — handle that, but with two well-known
problems: **you have to hand-pick the numeric/categorical balance**, and **high-cardinality
categorical columns quietly dominate the distance** regardless of whether they carry signal.

`mixdist` makes the weighting explicit and inspectable, scales past the `n × n` wall, and
hands the resulting geometry to ordinary Euclidean tooling.

```bash
pip install mixdist
```

Requires Python ≥ 3.9, NumPy and pandas. Nothing else.

---

## The problem, in one table

```python
from mixdist import MixedMetric, make_mixed_blobs

# 3 informative numeric + 2 informative categorical + 2 pure-noise categorical
X, y_true = make_mixed_blobs(n_samples=800, random_state=0)

MixedMetric(weights="equal").fit(X).column_report()
```

```
            kind  n_levels  expected_dissimilarity  weight  share
column
noise_0  nominal      30.0                   0.965     1.0  0.246
noise_1  nominal      30.0                   0.965     1.0  0.246
cat_1    nominal       3.0                   0.666     1.0  0.170
cat_0    nominal       3.0                   0.665     1.0  0.170
num_1    numeric       NaN                   0.246     1.0  0.063
num_2    numeric       NaN                   0.210     1.0  0.054
num_0    numeric       NaN                   0.200     1.0  0.051
```

Two columns of pure noise command **49 %** of the distance, while each informative
numeric column gets ~5 %. Equal weights equalise the *nominal* weight, not the
*contribution* — and contribution is what the clustering actually sees. The mechanism is
arithmetic, not bad luck: a balanced nominal column with `K` levels has expected
dissimilarity `1 − 1/K` (0.97 at 30 levels), while a range-scaled numeric column tops out
at 1/3 and is usually nearer 0.2.

`weights="balanced"` sets `w_k ∝ 1 / E[d_k]`, so every column contributes the same
expected distance:

```python
MixedMetric(weights="balanced").fit(X).column_report()["share"]
# every column ≈ 0.143
```

Measured over 5 seeds (800 rows, 3 clusters):

| weights | kNN purity (k=10) | ARI (KMeans on `transform`) | ARI s.d. |
|---|---|---|---|
| `"equal"` (Gower 1971) | 0.905 | 0.827 | 0.083 |
| `"type_balanced"` | 0.910 | 0.846 | 0.062 |
| `"balanced"` | **0.937** | **0.894** | **0.024** |

kNN purity — the fraction of a point's 10 nearest neighbours sharing its true cluster —
is reported because it measures the *metric* with no clustering algorithm in the way.
Note also the collapse in variance: `balanced` is not just better on average, it is three
times more stable across seeds.

(Reproduce with `python examples/weighting_matters.py`.)

---

## Quick start

```python
from mixdist import MixedMetric

metric = MixedMetric(weights="balanced").fit(df)

D = metric.pairwise(df)                        # full distance matrix
dist, idx = metric.kneighbors(n_neighbors=10)  # exact kNN, never builds n×n
Z = metric.transform(df)                       # ||Z_i − Z_j||² == Gower distance

metric.column_report()      # which variables own the distance
metric.explain(df, 3, 17)   # why rows 3 and 17 are similar, column by column
```

### Column typing

Types are inferred (`object`/`category`/`bool` → nominal, ordered `Categorical` →
ordinal, low-cardinality integers → nominal), and you override what matters:

```python
MixedMetric(
    categorical=["plan_tier", "region"],
    ordinal=["satisfaction"],
    numeric_range="robust",   # inner-quantile scaling; one outlier can't flatten a column
)
```

Missing values are handled the way Gower intended — the column drops out of that
*pair's* denominator, not the whole dataset.

---

## Scaling

Distances accumulate **one column at a time** into a bounded block, so no `(n, m, p)`
intermediate is ever built. Measured on this machine (7 columns; `python examples/scaling.py`):

| n | `kneighbors` | peak | `pairwise` | peak | reference `gower` | peak |
|---|---|---|---|---|---|---|
| 2,000 | 0.09 s | 164 MB | 0.06 s | 196 MB | 11.4 s | 18 MB |
| 5,000 | 0.52 s | 329 MB | 0.33 s | 528 MB | 71.1 s | 106 MB |
| 10,000 | 2.04 s | 330 MB | 1.31 s | 1,128 MB | 285.6 s | 412 MB |
| 20,000 | 7.96 s | **331 MB** | — | — | — | — |

Two things to read off it. **Speed**: ~200× faster than the reference implementation at
n = 10,000. **Memory**: `kneighbors` is flat in `n` — the matrix is never materialised —
while `pairwise` necessarily grows as `n²` because the matrix *is* the output. (`mixdist`
computes in float64 and `gower` in float32, which accounts for the 2× in the `pairwise`
column.)

So on anything large, don't ask for the matrix:

```python
# Top-k directly — O(n·k) memory, exact, no approximation
dist, idx = metric.kneighbors(query_df, reference_df, n_neighbors=20)

# Or stream the matrix in blocks
for start, stop, block in metric.iter_pairwise(df):
    ...
```

Results agree with the reference `gower` package to **4e-8** (float32 precision) — see
`tests/test_reference_gower.py`, which is run against it directly.

---

## Euclidean embedding (the interesting part)

Gower's distance is a non-negatively weighted average of `|u − v|` on range-scaled
numerics and `1[u ≠ v]` on nominals. Both are metrics of **negative type**, and negative
type is closed under non-negative combination — so `√d_Gower` embeds isometrically in
`ℓ₂`, and the map can be written down explicitly rather than recovered by MDS.

`transform()` builds it: thermometer features for numeric columns, scaled one-hot for
nominal ones, amplitudes chosen so that

```
‖transform(X)[i] − transform(X)[j]‖²  ==  pairwise(X)[i, j]
```

exactly, up to `1/n_bins` quantisation per numeric column.

```python
import faiss, numpy as np

Z = metric.transform(df, n_components=256)     # optional JL projection
index = faiss.IndexFlatL2(Z.shape[1])
index.add(np.ascontiguousarray(Z))
index.search(Z[:5], 10)                        # Gower nearest neighbours, at FAISS speed
```

The same trick makes `sklearn.cluster.KMeans`, `HDBSCAN` and UMAP operate on Gower
geometry without modification.

---

## Clustering

Two algorithms, chosen because they attack the weighting problem from opposite directions.

### `KPrototypes` — Huang (1997)

```python
from mixdist import KPrototypes

km = KPrototypes(n_clusters=4, gamma="modha-spangler", random_state=0).fit(df)
km.labels_, km.gamma_, km.cluster_centers_
```

`gamma="auto"` uses Huang's rule of thumb. `gamma="modha-spangler"` searches the balance
by minimising the product of the numeric and categorical within-cluster distortions
(Modha & Spangler, 2003) — no labels, no tuning, one run per grid point.

### `KAMILA` — Foss et al. (2016)

```python
from mixdist import KAMILA

km = KAMILA(n_clusters=4, random_state=0).fit(df)
km.cluster_centers_
km.level_probabilities_["plan_tier"]   # per-cluster level probabilities
```

KAMILA has **no** `gamma`. It models the continuous block as a spherical cluster with a
KDE-estimated radial density and the categorical block as per-cluster multinomials, then
compares log-densities — quantities already on a common scale. The balance comes from the
data.

This is the first Python implementation; the reference implementation is the
[R package `kamila`](https://github.com/ahfoss/kamila). It is written from the published
description, not ported, so behaviour agrees but numbers need not match bit for bit.

### Explaining clusters

```python
metric.explain_clusters(df, km.labels_)
```

```
          num_0   num_1   num_2   cat_0   cat_1  noise_0  noise_1
cluster
0        0.0110  0.0285  0.0219  0.0353  0.0338   0.0001   0.0001
1        0.0106  0.0258  0.0232  0.0314  0.0295   0.0002   0.0001
2        0.0135  0.0269  0.0241  0.0345  0.0307   0.0002   0.0002
```

Entry `(g, k)` is the weighted **dispersion reduction** of column `k` inside cluster `g`:
how much tighter the column gets once you condition on the cluster, on the Gower scale.
The noise columns score ~0.0002 against ~0.03 for the informative ones, and they do so
*even though they sit far from the global centre* — which is the trap this measure is
built to avoid. Negative entries mean the cluster is more heterogeneous than the data at
large. No surrogate model is involved.

---

## Design notes

- **No scikit-learn dependency**, but estimators implement `get_params`/`set_params`, so
  they drop into `Pipeline` and `GridSearchCV` when scikit-learn is present.
- **`gower_matrix(X)`** is provided as a one-line drop-in for the common
  `gower.gower_matrix` idiom, defaulting to `weights="equal"` for exact compatibility.
- **Constant columns are dropped** with a warning. They contribute no distance but do
  enter Gower's denominator, silently rescaling everything.

## Limitations

- `transform()` cannot represent missing values exactly — Gower's pair-dependent
  denominator has no single feature map. The default imputes and warns; pass
  `on_missing="error"` to refuse instead.
- `KAMILA`'s continuous model assumes spherical clusters, as in the original paper.
- `gamma="modha-spangler"` costs one full clustering per grid point.

## References

- Gower, J. C. (1971). *A general coefficient of similarity and some of its properties.* Biometrics 27(4).
- Huang, Z. (1997). *Clustering large data sets with mixed numeric and categorical values.* PAKDD.
- Modha, D. S. & Spangler, W. S. (2003). *Feature weighting in k-means clustering.* Machine Learning 52(3).
- Foss, A., Markatou, M., Ray, B. & Heching, A. (2016). *A semiparametric method for clustering mixed data.* Machine Learning 105(3).
- Schoenberg, I. J. (1938). *Metric spaces and positive definite functions.* TAMS 44(3) — the negative-type result behind `transform()`.

## License

BSD 3-Clause.
