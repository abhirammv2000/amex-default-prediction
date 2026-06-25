"""Plot the top-N LightGBM features by gain -> reports/figures/feature_importance.png.

Reads the importance table written by ``train_baseline.py``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def main(top_n: int = 25) -> None:
    imp = pd.read_csv(config.OUTPUT_DIR / "feature_importance.csv").head(top_n)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="#4C72B0")
    ax.set_title(f"Top {top_n} features by LightGBM gain")
    ax.set_xlabel("gain")
    fig.tight_layout()
    out = config.FIGURE_DIR / "feature_importance.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
