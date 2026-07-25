# mixdist

**数値・カテゴリ混合テーブルのための距離計算・近傍探索・クラスタリング**

*[English README](README.md)*

実務の顧客テーブルは `年齢` や `購入額` といった数値軸と、`性別` や `会員ランク` といった
カテゴリ軸が必ず混ざっています。標準的な答えである Gower 距離と k-prototypes はこれを
扱えますが、二つの既知の問題を抱えています。**数値とカテゴリのバランスを人手で決めなければ
ならない**こと、そして**高カーディナリティのカテゴリ列が、信号を持つかどうかに関係なく
距離を支配してしまう**ことです。

`mixdist` は重み付けを明示的かつ検査可能にし、`n × n` の壁を越えてスケールし、
得られた幾何を通常のユークリッド系ツールにそのまま渡せるようにします。

```bash
pip install mixdist
```

Python 3.9 以上、NumPy と pandas のみ。他の依存はありません。

---

## 問題を一枚の表で

```python
from mixdist import MixedMetric, make_mixed_blobs

# 情報を持つ数値3列 + 情報を持つカテゴリ2列 + 純粋なノイズのカテゴリ2列
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

**純粋なノイズ2列が距離の 49 % を占めています。**情報を持つ数値列は1列あたり約 5 % です。

これは運が悪いのではなく算術の帰結です。水準数 `K` のカテゴリ列の期待非類似度は `1 − 1/K`
（30水準なら 0.97）である一方、範囲正規化された数値列は最大でも 1/3、実際には 0.2 程度に
しかなりません。等重みは*名目上の重み*を揃えているだけで、*実際の寄与*は揃えていません。
そしてクラスタリングが見ているのは後者です。

`weights="balanced"` は `w_k ∝ 1 / E[d_k]` とすることで、全列の期待寄与を等しくします。

```python
MixedMetric(weights="balanced").fit(X).column_report()["share"]
# 全列が ≈ 0.143
```

5シードでの実測値（800行・3クラスタ）:

| weights | kNN純度 (k=10) | ARI (`transform` 上の KMeans) | ARI 標準偏差 |
|---|---|---|---|
| `"equal"`（Gower 1971） | 0.905 | 0.827 | 0.083 |
| `"type_balanced"` | 0.910 | 0.846 | 0.062 |
| `"balanced"` | **0.937** | **0.894** | **0.024** |

kNN純度（ある点の10近傍のうち、真のクラスタが一致する割合）を併記しているのは、
クラスタリングアルゴリズムを挟まずに*距離そのもの*を評価できるためです。
平均値の改善に加えて**標準偏差が 1/3 以下**になっている点にも注目してください。
`balanced` は平均的に良いだけでなく、シード間で3倍安定しています。

（再現: `python examples/weighting_matters.py`）

---

## クイックスタート

```python
from mixdist import MixedMetric

metric = MixedMetric(weights="balanced").fit(df)

D = metric.pairwise(df)                        # 全距離行列
dist, idx = metric.kneighbors(n_neighbors=10)  # 厳密kNN、n×n を作らない
Z = metric.transform(df)                       # ‖Z_i − Z_j‖² == Gower距離

metric.column_report()      # どの変数が距離を握っているか
metric.explain(df, 3, 17)   # 3行目と17行目がなぜ似ているか、列ごとに
```

### 列の型付け

型は自動推論されます（`object`/`category`/`bool` → 名義、順序付き `Categorical` → 順序、
低カーディナリティ整数 → 名義）。重要な列は明示的に上書きしてください。

```python
MixedMetric(
    categorical=["plan_tier", "region"],
    ordinal=["satisfaction"],
    numeric_range="robust",   # 内側分位点でスケーリング。外れ値1点で列が潰れない
)
```

欠損値は Gower の本来の定義どおりに扱われます。データセット全体からではなく、
**その「ペア」の分母からその列が抜ける**という処理です。

---

## スケーラビリティ

距離は**1列ずつ**有界なブロックに累積されるため、`(n, m, p)` の中間配列を一度も作りません。
実測値（7列、`python examples/scaling.py`）:

| n | `kneighbors` | ピーク | `pairwise` | ピーク | 参照実装 `gower` | ピーク |
|---|---|---|---|---|---|---|
| 2,000 | 0.09 s | 164 MB | 0.06 s | 196 MB | 11.4 s | 18 MB |
| 5,000 | 0.52 s | 329 MB | 0.33 s | 528 MB | 71.1 s | 106 MB |
| 10,000 | 2.04 s | 330 MB | 1.31 s | 1,128 MB | 285.6 s | 412 MB |
| 20,000 | 7.96 s | **331 MB** | — | — | — | — |

読み取るべきは二点です。**速度**: n = 10,000 で参照実装の約 200 倍。
**メモリ**: `kneighbors` は n に対して一定（行列を作らない）である一方、
`pairwise` は行列そのものが出力なので必然的に `n²` で増えます。
（`mixdist` は float64、`gower` は float32 で計算しており、`pairwise` 列の2倍差はこれに由来します。）

したがって大規模データでは行列を要求しないでください。

```python
# top-k を直接 — O(n·k) メモリ、厳密解、近似なし
dist, idx = metric.kneighbors(query_df, reference_df, n_neighbors=20)

# あるいは行列をブロック単位でストリーム
for start, stop, block in metric.iter_pairwise(df):
    ...
```

結果は参照実装 `gower` パッケージと **4e-8**（float32 の精度限界）で一致します。
`tests/test_reference_gower.py` で直接突き合わせています。

---

## ユークリッド埋め込み（技術的な要点）

Gower 距離は、範囲正規化された数値上の `|u − v|` と、名義変数上の `1[u ≠ v]` の、
非負重み付き平均です。**この二つはいずれも負値型（negative type）の距離**であり、
負値型は非負線形結合について閉じています。したがって `√d_Gower` は `ℓ₂` に等長埋め込み
でき、しかもその写像は MDS で数値的に復元するのではなく**明示的に書き下せます**。

`transform()` はそれを構成します。数値列にはサーモメータ符号化、名義列にはスケール済み
one-hot を割り当て、振幅を適切に選ぶことで

```
‖transform(X)[i] − transform(X)[j]‖²  ==  pairwise(X)[i, j]
```

が成立します（数値列あたり `1/n_bins` の量子化誤差を除いて厳密）。

```python
import faiss, numpy as np

Z = metric.transform(df, n_components=256)     # 任意でJL射影
index = faiss.IndexFlatL2(Z.shape[1])
index.add(np.ascontiguousarray(Z))
index.search(Z[:5], 10)                        # Gower近傍を FAISS の速度で
```

同じ仕掛けにより、`sklearn.cluster.KMeans`、`HDBSCAN`、UMAP が無改造で
Gower 幾何の上で動作します。

---

## クラスタリング

重み付け問題に正反対の方向から挑む2手法を実装しています。

### `KPrototypes` — Huang (1997)

```python
from mixdist import KPrototypes

km = KPrototypes(n_clusters=4, gamma="modha-spangler", random_state=0).fit(df)
km.labels_, km.gamma_, km.cluster_centers_
```

`gamma="auto"` は Huang の経験則を使います。`gamma="modha-spangler"` は、数値側と
カテゴリ側のクラスタ内歪みの**積**を最小化することでバランスを探索します
（Modha & Spangler, 2003）。ラベル不要・チューニング不要で、グリッド1点につき1回の実行です。

### `KAMILA` — Foss et al. (2016)

```python
from mixdist import KAMILA

km = KAMILA(n_clusters=4, random_state=0).fit(df)
km.cluster_centers_
km.level_probabilities_["plan_tier"]   # クラスタごとの水準確率
```

KAMILA には `gamma` が**ありません**。連続部分を球対称クラスタとしてモデル化し、
その動径密度を KDE で推定、カテゴリ部分をクラスタごとの多項分布としてモデル化した上で、
**対数密度どうしを比較**します。対数密度は最初から共通尺度に乗っているため、
バランスはデータ自身が決めます。

これは **Python では初の実装**です。参照実装は
[R パッケージ `kamila`](https://github.com/ahfoss/kamila) です。
published な記述からの実装であり移植ではないため、挙動は一致しますが
数値がビット単位で一致することは保証しません。

### クラスタの説明

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

`(g, k)` 成分は、クラスタ `g` における列 `k` の重み付き**分散減少量**です。
すなわち「そのクラスタに条件付けたとき、その列がどれだけ締まるか」を Gower 尺度で
表したものです。ノイズ列は約 0.0002、情報を持つ列は約 0.03 と2桁の差がつきます。
しかも**ノイズ列は全体中心から遠く離れているにもかかわらず** 0 に近い値になります。
これがこの指標が回避するために設計された罠です。負の値は、そのクラスタが
データ全体よりも不均質であることを意味します。代理モデルは一切使っていません。

---

## 設計上の方針

- **scikit-learn に依存しません**が、`get_params`/`set_params` を実装しているため、
  scikit-learn が入っていれば `Pipeline` や `GridSearchCV` にそのまま乗ります。
- **`gower_matrix(X)`** を、よくある `gower.gower_matrix` 呼び出しの1行代替として
  提供しています。既定は `weights="equal"` で、完全な互換性を保ちます。
- **定数列は警告付きで除外されます。** 距離には寄与しないのに Gower の分母には入るため、
  放置すると全距離が静かにリスケールされてしまいます。

## 制約

- `transform()` は欠損値を厳密に表現できません。Gower のペア依存の分母は
  単一の特徴写像を持たないためです。既定では補完して警告を出します。
  拒否させたい場合は `on_missing="error"` を指定してください。
- `KAMILA` の連続部分のモデルは、原論文どおり球対称クラスタを仮定します。
- `gamma="modha-spangler"` はグリッド1点につきクラスタリング1回のコストがかかります。

## 参考文献

- Gower, J. C. (1971). *A general coefficient of similarity and some of its properties.* Biometrics 27(4).
- Huang, Z. (1997). *Clustering large data sets with mixed numeric and categorical values.* PAKDD.
- Modha, D. S. & Spangler, W. S. (2003). *Feature weighting in k-means clustering.* Machine Learning 52(3).
- Foss, A., Markatou, M., Ray, B. & Heching, A. (2016). *A semiparametric method for clustering mixed data.* Machine Learning 105(3).
- Schoenberg, I. J. (1938). *Metric spaces and positive definite functions.* TAMS 44(3) — `transform()` の根拠となる負値型の結果。

## ライセンス

BSD 3-Clause
