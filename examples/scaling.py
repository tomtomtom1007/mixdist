"""How mixdist behaves as the table grows.

Measures three things:

* ``pairwise``   — the full matrix. Inherently O(n²) in time *and* memory.
* ``kneighbors`` — exact top-k. O(n²) time, but **O(n·k) memory**: the matrix is
  never materialised, which is the difference between "works" and "MemoryError"
  on a real customer table.
* the reference ``gower`` package, when installed, for context.

Run:  python examples/scaling.py
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd

from mixdist import MixedMetric, make_mixed_blobs

SIZES = (2_000, 5_000, 10_000, 20_000)


def timed(fn):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return result, elapsed, peak


def try_reference(frame: pd.DataFrame):
    try:
        import gower
    except ImportError:
        return None, None, None
    obj = frame.copy()
    for name in obj.columns:
        if obj[name].dtype != float:
            obj[name] = obj[name].astype(object)
    try:
        return timed(lambda: gower.gower_matrix(obj))
    except (TypeError, ValueError) as exc:
        print(f"    reference gower unavailable on this pandas: {exc}")
        return None, None, None


def main() -> None:
    rows = []
    for n in SIZES:
        X, _ = make_mixed_blobs(n_samples=n, random_state=0)
        metric = MixedMetric(weights="equal").fit(X)

        _, t_knn, m_knn = timed(lambda m=metric: m.kneighbors(n_neighbors=10))
        row = {"n": n, "knn_s": t_knn, "knn_MB": m_knn}

        if n <= 10_000:  # the full matrix stops being reasonable beyond this
            D, t_full, m_full = timed(lambda m=metric, f=X: m.pairwise(f))
            row |= {"pairwise_s": t_full, "pairwise_MB": m_full}

            G, t_ref, m_ref = try_reference(X)
            if G is not None:
                row |= {
                    "gower_s": t_ref,
                    "gower_MB": m_ref,
                    "max_abs_diff": float(np.abs(D - G).max()),
                }
        rows.append(row)

    table = pd.DataFrame(rows).set_index("n")
    formats = {c: "{:.3f}".format for c in table.columns if c != "max_abs_diff"}
    if "max_abs_diff" in table:
        formats["max_abs_diff"] = "{:.2e}".format
    print("\nmixdist scaling (7 columns: 3 numeric, 4 nominal)\n")
    print(table.to_string(formatters=formats, na_rep="-"))
    print(
        "\nknn_MB is flat in n: kneighbors streams blocks and keeps only the top k.\n"
        "pairwise_MB grows as n² because the matrix itself is the output."
    )


if __name__ == "__main__":
    main()
