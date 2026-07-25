"""Does the weighting scheme actually matter?

Reproduces the table in the README.  Two measures are reported because they
answer different questions:

``knn_purity``
    Fraction of a point's 10 nearest neighbours that share its true cluster.
    This measures the *metric* directly, with no clustering algorithm in the
    way, so it is the honest way to compare weighting schemes.

``ari_kmeans``
    Adjusted Rand Index of KMeans run on ``metric.transform(X)``.  Because the
    embedding is exact, this is KMeans operating on Gower geometry.

Requires scikit-learn:  pip install mixdist[test]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from mixdist import MixedMetric, make_mixed_blobs

SCHEMES = ("equal", "type_balanced", "balanced")
N_SEEDS = 5


def evaluate(scheme: str, X, y) -> tuple[float, float]:
    metric = MixedMetric(weights=scheme).fit(X)
    _, idx = metric.kneighbors(n_neighbors=10)
    purity = float((y[idx] == y[:, None]).mean())

    Z = metric.transform(X, dtype=np.float64)
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Z)
    return purity, float(adjusted_rand_score(y, labels))


def main() -> None:
    rows = []
    for scheme in SCHEMES:
        scores = np.array(
            [evaluate(scheme, *make_mixed_blobs(n_samples=800, random_state=s))
             for s in range(N_SEEDS)]
        )
        rows.append(
            {
                "weights": scheme,
                "knn_purity": scores[:, 0].mean(),
                "ari_kmeans": scores[:, 1].mean(),
                "ari_sd": scores[:, 1].std(),
            }
        )

    table = pd.DataFrame(rows).set_index("weights").round(3)
    print(f"\nmean over {N_SEEDS} seeds, 800 rows, 3 clusters")
    print("3 informative numeric + 2 informative nominal + 2 noise nominal (30 levels)\n")
    print(table.to_string())

    print("\nWhere the distance goes under equal weights:\n")
    X, _ = make_mixed_blobs(n_samples=800, random_state=0)
    print(MixedMetric(weights="equal").fit(X).column_report().round(3).to_string())


if __name__ == "__main__":
    main()
