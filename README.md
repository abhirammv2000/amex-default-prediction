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
feature matrix (458K × 1,628)
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
* **Ensembling:** a second family (**XGBoost**) is trained on the *same* fold
  splits so its OOF aligns row-for-row, then [`blend.py`](src/blend.py) picks the
  LGB/XGB weight that maximises the Amex metric **on OOF** (not the test set) and
  applies it to the test predictions — see iterations v4a/v4b below.

Key hyper-parameters (see [`train_baseline.py`](src/train_baseline.py)):
`learning_rate=0.03`, `num_leaves=128`, `feature_fraction=0.4`,
`bagging_fraction=0.8`, `lambda_l2=2.0`.

### Scaling out: cloud training

Training the wide (1,628-feature) model is memory-bound on a 16 GB laptop, so
the training step is offloaded to a **GCP spot VM** (`n2-highmem-8`, 8 vCPU /
64 GB) — see [`cloud/`](cloud/). The workflow is fully unattended and
fault-tolerant:

* [`cloud/launch.sh`](cloud/launch.sh) enables the APIs, uploads code + the
  feature table to a GCS bucket, and creates a **spot** VM (~$0.10–0.15/hr).
* [`cloud/bootstrap_vm.sh`](cloud/bootstrap_vm.sh) runs as the VM **startup
  script**: it installs the stack, pulls the data, trains, pushes the models +
  CV metadata back to GCS with a `_STATUS` marker, and **powers the VM off** so
  billing stops automatically.

Because the job is driven by the VM (not the local session) and signals
completion through GCS, the run survives laptop sleeps / disconnects, and the
whole 5-fold training completes in ~10–15 minutes for a few cents.

## 6. Results

The project is developed as a series of cross-validated iterations. Every score
below is the **out-of-fold (OOF) Amex metric** under the same 5-fold
StratifiedKFold split — an honest, leakage-free estimate.

| Iteration | Model / change | OOF Amex | Δ vs prev best |
|-----------|----------------|---------:|---------------:|
| **v1** — baseline aggregations (`mean/std/min/max/last`), 920 feats | LightGBM | 0.79123 | — |
| **v2** — + trend/deviation features (`first`, `last−mean`, `last−first`, `range`), 1,628 feats | LightGBM | **0.79266** | **+0.00143** |
| **v3** — Optuna hyper-parameter search (9 trials) | LightGBM | 0.79247 | −0.00019 |
| **v4a** — second model family | XGBoost | 0.79064 | — |
| **v4b** — weighted blend (0.86·LGB + 0.14·XGB) | **LGB + XGB** | **0.79294** | **+0.00028** |

For context, the competition's **private-leaderboard winners scored ≈ 0.808**,
and a single LightGBM on aggregated features typically lands around
**0.78–0.79** — so this model is already competitive, and the CV is tight
(std < 0.004), meaning the estimate is stable across folds.

**Reading the iterations honestly:**
* **v2** (trend features) was the biggest lever — capturing how a customer's
  account is *changing*, not just its level. **18 of the top-100 features by gain
  are the new derived features.**
* **v3** (tuning) **did not improve** the score: the Optuna search only completed
  9 trials inside its time budget (each trial is slow on 1,628 features), and the
  hand-chosen baseline params were already near-optimal. A negative result, kept
  in the table for transparency.
* **v4** (blend) gave a small but real lift. XGBoost is the weaker solo model, but
  it is **decorrelated** enough from LightGBM that a 0.86/0.14 blend — with the
  weight chosen on OOF, not the test set — beats either alone.

**Top features by gain** (v2, full table in `outputs/feature_importance.csv`):

| Rank | Feature | Meaning |
|-----:|---------|---------|
| 1 | `B_9_last`  | latest balance state |
| 2 | `P_2_last`  | most recent payment feature |
| 3 | `P_2_mean`  | average payment level |
| 4 | `P_2_max` / `P_2_min` | payment range over history |
| 6 | `D_48_last` | latest delinquency state |
| 13 | `P_2_first` | earliest payment level (new trend feature) |

`P_2` dominates across its aggregations, and `*_last` features confirm that a
customer's **most recent statement** carries the most predictive signal.

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
├── cloud/
│   ├── launch.sh                   # provision GCP spot VM + upload via GCS
│   └── bootstrap_vm.sh             # VM startup-script: train + push results
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
    ├── tune.py                     # Optuna search + tuned 5-fold retrain
    ├── train_xgb.py                # 5-fold XGBoost (aligned folds for blending)
    ├── blend.py                    # OOF-optimal LGB+XGB blend -> submission
    ├── reconstruct_oof_xgb.py      # rebuild XGB OOF from saved fold models
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

# 5. (optional) second model + ensemble
python src/tune.py --n-trials 50          # Optuna-tuned LightGBM
python src/train_xgb.py                    # XGBoost on the same folds
python src/blend.py                        # OOF-optimal blend -> submission_blend.csv

# 6. (optional) EDA figures / single-model submission
python src/eda.py
python src/predict.py
```

> **Heavy steps run on the cloud.** Training the 1,628-feature models is
> memory-bound on a 16 GB laptop, so they are launched on a GCP VM, e.g.:
> ```bash
> JOB=xgb MACHINE=n2-highmem-8 PROVISIONING=STANDARD \
>   RUNCMD="python3 -u train_xgb.py" bash cloud/launch.sh
> ```

## 9. Next steps

Done so far: trend/deviation features ✓, Optuna tuning ✓ (no gain — see v3),
XGBoost + blend ✓. Promising directions from here:

* **More features:** lag-1 deltas, per-feature slopes, "round-number" payment
  flags, and `after-pay` features (balance − payment) that ranked highly in
  winning solutions.
* **CatBoost** as a third blend partner (handles the categoricals natively) and a
  **GRU/Transformer** over the raw monthly sequence for true model diversity.
* **A longer tuning budget** — v3 only fit 9 Optuna trials; a multi-hour search
  (or tuning on a single fold at lower learning rate) may yet find gains.
* **`dart` boosting** and **knowledge-distillation / pseudo-labelling** on the
  large test set.

## 10. Acknowledgements

* American Express & Kaggle for the competition and dataset.
* The Kaggle community for the public reference metric implementation.
