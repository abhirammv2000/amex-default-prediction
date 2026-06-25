# American Express — Default Prediction

Predicting the probability that a credit-card customer will **default** on their
balance, using 13 months of anonymized monthly statement data. This repository
implements an end-to-end, memory-efficient machine-learning pipeline for the
[Kaggle *American Express - Default Prediction*](https://www.kaggle.com/competitions/amex-default-prediction)
competition and reports a strong, fully cross-validated **LightGBM baseline**.

> **Status:** Baseline complete (data pipeline → features → CV model → submission).
> Deployment/serving is intentionally out of scope for this stage.

### Highlights

* **Engineered a memory-safe pipeline** that processes **48 GB of raw CSVs on a
  16 GB-RAM machine** via chunked streaming, `float32` Parquet conversion, and
  column-batched aggregation — nothing is ever fully loaded into memory.
* **Faithful, unit-tested implementation** of the competition's custom
  rank metric (normalized Gini + default-capture@4 %, with ×20 negative
  weighting).
* **Cross-validated LightGBM baseline scoring 0.79123** (5-fold OOF Amex
  metric) — competitive with strong public baselines (winners ≈ 0.808).
* Clean, reproducible, scripted pipeline: profile → convert → features →
  train → predict, plus an EDA notebook.

**Tech:** Python · pandas · PyArrow · LightGBM · scikit-learn · matplotlib

---

## 1. Problem statement

American Express asked competitors to predict, for each `customer_ID`, the
probability of a **future payment default** (`target = 1`) given the customer's
recent statement history. This is the core of consumer-credit risk management:
a better default model means fewer losses from bad loans **and** fewer good
customers wrongly declined.

* **Task:** binary classification → output a default *probability* per customer.
* **Label definition (Amex):** a default is "no payment within 120 days after
  the latest statement", observed over an 18-month performance window.
* **Granularity:** the model produces **one prediction per customer**, but each
  customer is described by **up to 13 monthly statements** (a short
  multivariate time series).

## 2. Data

| File | Rows | Size | Description |
|------|------|------|-------------|
| `train_data.csv`        | ~5.53 M statements | **15.6 GB** | training statements |
| `test_data.csv`         | ~11.4 M statements | **32.3 GB** | test statements |
| `train_labels.csv`      | ~458 K customers   | 30 MB | binary `target` per customer |
| `sample_submission.csv` | ~924 K customers   | 60 MB | submission format |

* **190 columns:** `customer_ID`, `S_2` (statement date), and **188 anonymized
  features**.
* Features are grouped by an informative prefix:

  | Prefix | Meaning | # features |
  |--------|---------|-----------:|
  | `D_*`  | Delinquency | 96 |
  | `B_*`  | Balance     | 40 |
  | `R_*`  | Risk        | 28 |
  | `S_*`  | Spend       | 21 |
  | `P_*`  | Payment     | 3  |

* **11 categorical features:** `B_30, B_38, D_114, D_116, D_117, D_120, D_126,
  D_63, D_64, D_66, D_68` (`D_63`/`D_64` are strings; the rest are integer codes).
* Statements span **Mar 2017 → Mar 2018**; most customers have the full 13
  monthly statements (median = 13), but some have as few as 1.
* The public data is **noisy/quantized** by design and ~30 features are
  >50 % missing — handled naturally by gradient-boosted trees.

> Raw data is **not** committed (see `.gitignore`). Download it from the
> competition page into `amex-default-prediction/`.

## 3. Evaluation metric

The competition uses a custom rank metric **M = 0.5 · (G + D)**:

* **G** — *normalized Gini coefficient* (overall ranking quality).
* **D** — *default rate captured at 4 %*: the share of true defaults that fall
  in the top-ranked 4 % of predictions (a recall/sensitivity statistic).

For **both** components the **negative class is weighted ×20** to undo the 5 %
negative down-sampling applied to the public data. Maximum score = 1.0.

A faithful, unit-tested re-implementation lives in
[`src/metric.py`](src/metric.py) — both a readable pandas version and a fast
NumPy version (used for early stopping), with a self-test confirming they agree
and that a perfect ranking yields normalized Gini = 1.0.

## 4. The memory challenge & pipeline design

The raw CSVs (**48 GB** combined) are far larger than the **16 GB** of RAM on
the development machine, so nothing can be `read_csv`'d in one shot. The pipeline
is built around three memory-safe ideas:

1. **Stream → Parquet, downcast to `float32`.**
   [`convert_to_parquet.py`](src/convert_to_parquet.py) reads the CSVs in
   500 K-row chunks and appends to a single Parquet file via a `ParquetWriter`,
   casting the 185 numeric columns to `float32`. This cuts size **~4×**
   (15.6 GB → ~3.4 GB) and makes every later read fast and columnar.

2. **Collapse the time series → one row per customer.**
   [`feature_engineering.py`](src/feature_engineering.py) aggregates each
   customer's statements into summary statistics:
   * numeric (177): `mean, std, min, max, last`
   * categorical (11): `last, nunique, count`
   * plus `statement_count` and `history_days`.
   Columns are read from Parquet in **small batches** (40 at a time) so the full
   5.5 M-row table is never in memory at once; `customer_ID` is factorized to
   integer codes **once** and reused for every batch.

3. **Train on the compact feature matrix.** ~458 K customers × ~920 features
   (`float32`, ≈ 1.6 GB) fits comfortably in memory for 5-fold LightGBM.

```
raw CSV (48 GB)
  │  convert_to_parquet.py   (chunked, float32)
  ▼
Parquet (~10 GB)
  │  feature_engineering.py  (per-customer aggregation, column-batched)
  ▼
feature matrix (458K × ~920)
  │  train_baseline.py       (5-fold stratified CV, LightGBM)
  ▼
models + OOF + CV score
  │  predict.py              (average fold models)
  ▼
submission.csv
```

## 5. Modelling

* **Model:** LightGBM (`binary` objective), 5-fold **StratifiedKFold** CV.
* **Early stopping** on the **official Amex metric** (not log-loss), so the model
  is optimized for what actually scores.
* **Categorical features** passed natively to LightGBM.
* Out-of-fold (OOF) predictions give an honest CV estimate; test predictions
  average the five fold models.

Key hyper-parameters (see [`train_baseline.py`](src/train_baseline.py)):
`learning_rate=0.03`, `num_leaves=128`, `feature_fraction=0.4`,
`bagging_fraction=0.8`, `lambda_l2=2.0`.

## 6. Results

**5-fold cross-validated LightGBM baseline:**

| Metric | Value |
|--------|------:|
| **OOF Amex metric** | **0.79123** |
| Mean fold Amex | 0.79159 |
| Std across folds | 0.00376 |
| Per-fold | 0.79777 / 0.78899 / 0.79096 / 0.78688 / 0.79335 |
| Features | 920 |
| Customers (train) | 458,913 |

For context, the competition's **private-leaderboard winners scored ≈ 0.808**,
and a single well-tuned LightGBM on aggregated features typically lands around
**0.78–0.79** — so this baseline is already competitive, and the CV is tight
(std < 0.004), meaning the estimate is stable across folds.

**Top features by gain** (full table in `outputs/feature_importance.csv`):

| Rank | Feature | Meaning |
|-----:|---------|---------|
| 1 | `P_2_last`  | most recent payment feature |
| 2 | `P_2_min`   | worst payment over history |
| 3 | `P_2_mean`  | average payment level |
| 4 | `D_48_last` | latest delinquency state |
| 5 | `D_44_last` | latest delinquency state |
| 6 | `B_9_last`  | latest balance state |
| 7 | `B_11_last` | latest balance state |

`P_2` dominates by a wide margin — its `last`/`min`/`mean` aggregations are the
three strongest signals, which matches the well-documented behaviour of this
dataset. The prevalence of `*_last` features confirms that a customer's **most
recent statement** carries the most predictive information.

<p align="center">
  <img src="reports/figures/target_distribution.png" width="45%" alt="Target distribution">
  <img src="reports/figures/p2_by_target.png" width="45%" alt="P_2 by target">
</p>
<p align="center">
  <img src="reports/figures/feature_importance.png" width="60%" alt="Top features by gain">
</p>

## 7. Repository layout

```
amex/
├── README.md
├── requirements.txt
├── .gitignore
├── amex-default-prediction/        # raw Kaggle CSVs (gitignored)
├── data/processed/                 # parquet + engineered features (gitignored)
├── notebooks/
│   └── 01_eda.ipynb                # exploratory data analysis
├── reports/figures/                # committed EDA + importance plots (README)
├── outputs/                        # large regenerable artifacts (gitignored)
│   ├── models/                     # per-fold LightGBM models + cv_metadata.json
│   └── submissions/                # submission CSVs
└── src/
    ├── config.py                   # paths, column lists, constants
    ├── metric.py                   # official Amex metric (+ self-test)
    ├── profile_data.py             # quick sample profiling + metric test
    ├── eda.py                      # generates EDA figures
    ├── plot_importance.py          # feature-importance plot
    ├── convert_to_parquet.py       # CSV -> downcast Parquet (chunked)
    ├── feature_engineering.py      # per-customer aggregation
    ├── train_baseline.py           # 5-fold LightGBM CV
    └── predict.py                  # build submission from fold models
```

## 8. How to run

```bash
# 0. environment
conda activate amex-prediction          # or: pip install -r requirements.txt

# 1. sanity-check the metric + profile a data sample
python src/profile_data.py --rows 300000

# 2. convert the giant CSVs to Parquet (one-time, ~minutes)
python src/convert_to_parquet.py --which both

# 3. build per-customer features
python src/feature_engineering.py --which both

# 4. train the 5-fold LightGBM baseline (prints CV Amex score)
python src/train_baseline.py

# 5. (optional) EDA figures
python src/eda.py

# 6. generate the submission
python src/predict.py
```

## 9. Next steps (beyond the baseline)

* **Richer features:** last−mean / last−first diffs, lag-1 deltas, per-feature
  trends/slopes, "round-number" payment flags, and `after-pay` features
  (balance minus payment) that ranked highly in winning solutions.
* **Model diversity:** XGBoost + CatBoost + a GRU/Transformer over the raw
  monthly sequence, then blend.
* **`dart` boosting** and Optuna hyper-parameter search.
* **Knowledge-distillation / pseudo-labelling** on the large test set.

## 10. Acknowledgements

* American Express & Kaggle for the competition and dataset.
* The Kaggle community for the public reference metric implementation.
