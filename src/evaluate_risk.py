"""Credit-risk evaluation of the model's out-of-fold (OOF) predictions.

The competition metric only rewards *rank ordering*. A model used for real
lending decisions also has to be **calibrated** (its score must behave like a
true probability of default for pricing / loss provisioning) and has to be
judged on **business terms** (KS, capture by score band, the approval-rate vs
bad-rate trade-off). This script reports all of that from the OOF predictions,
which are an honest, leakage-free hold-out (each row scored by models not
trained on it).

Outputs: reports/figures/{calibration,score_bands,approval_tradeoff}.png and a
printed report; reports/risk_report.md.

Usage:
    python evaluate_risk.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import config
from metric import amex_metric_np

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative good/bad score CDFs."""
    order = np.argsort(p)
    y = y[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def two_fold_isotonic(y, p, seed=config.SEED):
    """Honestly-calibrated probabilities: fit isotonic on one half, apply to the
    other, and swap — so calibration is never evaluated on its own fit data."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    a, b = idx[:half], idx[half:]
    out = np.empty_like(p, dtype=float)
    for fit, app in ((a, b), (b, a)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p[fit], y[fit])
        out[app] = iso.predict(p[app])
    return out


def score_band_table(y, p, n=10) -> pd.DataFrame:
    """Decile table with band 1 = highest predicted risk; shows the bad-rate per
    band and the cumulative share of all defaults captured down to that band."""
    df = pd.DataFrame({"y": y, "p": p})
    # qcut bins p ascending; reverse the labels so band 1 = highest risk.
    df["band"] = pd.qcut(df["p"].rank(method="first"), n,
                         labels=range(n, 0, -1)).astype(int)
    g = df.groupby("band").agg(
        customers=("y", "size"), defaults=("y", "sum"),
        avg_pred=("p", "mean"), bad_rate=("y", "mean")).sort_index()
    g["cum_defaults_%"] = (g["defaults"].cumsum() / df["y"].sum() * 100).round(1)
    g["bad_rate"] = (g["bad_rate"] * 100).round(1)
    g["avg_pred"] = (g["avg_pred"] * 100).round(1)
    return g


def main() -> None:
    oof = pd.read_parquet(config.PROCESSED_DIR / "oof_predictions.parquet")
    y = oof["target"].values.astype(int)
    p = oof["oof_pred"].values
    fig_dir = config.FIGURE_DIR

    # ---- headline metrics ---------------------------------------------------
    auc = roc_auc_score(y, p)
    gini = 2 * auc - 1
    ks = ks_statistic(y, p)
    amex = amex_metric_np(y, p)
    brier_raw = brier_score_loss(y, p)
    p_cal = two_fold_isotonic(y, p)
    brier_cal = brier_score_loss(y, p_cal)

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 56)
    out("CREDIT-RISK EVALUATION (out-of-fold, 458,913 customers)")
    out("=" * 56)
    out(f"Amex metric        : {amex:.5f}")
    out(f"AUC / Gini         : {auc:.5f} / {gini:.5f}")
    out(f"KS statistic       : {ks:.5f}  (separation of goods vs bads)")
    out(f"Default rate       : {y.mean():.4f}")
    out(f"Brier  (raw)       : {brier_raw:.5f}")
    out(f"Brier  (isotonic)  : {brier_cal:.5f}  "
        f"({'better' if brier_cal < brier_raw else 'worse'})")

    # ---- calibration curve --------------------------------------------------
    frac_raw, mean_raw = calibration_curve(y, p, n_bins=20, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y, p_cal, n_bins=20, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(mean_raw, frac_raw, "o-", color="#C44E52", label=f"raw (Brier {brier_raw:.4f})")
    ax.plot(mean_cal, frac_cal, "s-", color="#4C72B0", label=f"isotonic (Brier {brier_cal:.4f})")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed default rate")
    ax.set_title("Calibration (reliability) curve"); ax.legend()
    fig.savefig(fig_dir / "calibration.png"); plt.close(fig)

    # ---- score bands --------------------------------------------------------
    bands = score_band_table(y, p)
    out("\nScore bands (decile 1 = highest predicted risk):")
    out(bands.to_string())
    top_decile_capture = bands["cum_defaults_%"].iloc[0]
    out(f"\nTop decile captures {top_decile_capture:.1f}% of all defaults.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bands.index, bands["bad_rate"], color="#C44E52", alpha=0.85)
    ax.set_xlabel("risk decile (1 = riskiest)"); ax.set_ylabel("actual default rate (%)")
    ax.set_title("Default rate by predicted-risk decile")
    fig.savefig(fig_dir / "score_bands.png"); plt.close(fig)

    # ---- approval-rate vs bad-rate trade-off --------------------------------
    order = np.argsort(p)              # approve lowest risk first
    y_sorted = y[order]
    approval = np.arange(1, len(y) + 1) / len(y)
    bad_rate_approved = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(approval * 100, bad_rate_approved * 100, color="#4C72B0")
    ax.axhline(y.mean() * 100, color="gray", ls="--", lw=1,
               label=f"portfolio avg {y.mean()*100:.1f}%")
    for a in (0.5, 0.7, 0.9):
        i = int(a * len(y)) - 1
        ax.annotate(f"{a*100:.0f}% approved\n{bad_rate_approved[i]*100:.1f}% bad",
                    (a * 100, bad_rate_approved[i] * 100),
                    textcoords="offset points", xytext=(-10, 10), fontsize=8)
    ax.set_xlabel("approval rate (%)"); ax.set_ylabel("bad rate among approved (%)")
    ax.set_title("Approval-rate vs bad-rate trade-off"); ax.legend()
    fig.savefig(fig_dir / "approval_tradeoff.png"); plt.close(fig)

    out("\nSaved figures: calibration.png, score_bands.png, approval_tradeoff.png")
    (config.REPORTS_DIR / "risk_report.md").write_text(
        "# Credit-risk evaluation report\n\n```\n" + "\n".join(lines) + "\n```\n")


if __name__ == "__main__":
    main()
