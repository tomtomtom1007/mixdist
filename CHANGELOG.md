# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-26

First release. [PyPI](https://pypi.org/project/mixdist/0.1.0/)

### Added

- `MixedMetric` — a fitted, inspectable Gower-family distance over mixed-type tables.
  - `pairwise`, `iter_pairwise` (block-streamed) and `kneighbors` (exact top-k, memory
    flat in `n`).
  - `column_report()` — per-column expected dissimilarity, weight, and share of the
    distance actually commanded.
  - `explain()` / `explain_pairs()` — exact additive per-column decomposition of a
    pairwise distance.
  - `explain_clusters()` — weighted dispersion reduction per column and cluster.
- Weighting schemes `"equal"` (Gower 1971), `"balanced"` (equal expected contribution per
  column) and `"type_balanced"`, plus explicit mappings and sequences.
- `transform()` — an exact Euclidean feature map for the Gower geometry via thermometer
  and scaled one-hot encoding, so FAISS, hnswlib, UMAP and `KMeans` operate on Gower
  distances unmodified. Optional Johnson–Lindenstrauss projection via `n_components`.
- `KPrototypes` (Huang, 1997) with `gamma="auto"` and `gamma="modha-spangler"`.
- `KAMILA` (Foss et al., 2016) — first Python implementation; no numeric/categorical
  trade-off parameter.
- `Schema` — column-type inference with explicit overrides, ordinal support, robust
  numeric scaling, constant-column dropping, and Gower-correct missing-value handling.
- `make_mixed_blobs` — synthetic mixed-type data including high-cardinality noise columns.
- `gower_matrix()` — one-call drop-in for the common `gower.gower_matrix` idiom.

### Validated against

- The reference [`gower`](https://pypi.org/project/gower/) package, to 4e-8
  (`tests/test_reference_gower.py`).

[0.1.0]: https://github.com/tomtomtom1007/mixdist/releases/tag/v0.1.0
