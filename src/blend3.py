"""Final 3-way blend: LightGBM + XGBoost + GRU.

Finds the weights that maximise the Amex metric on the aligned OOF predictions of
the three models, then applies them to the test predictions to write the final
submission.

Test predictions reuse what already exists (no recompute):
  * LGB test  = submission_lgbm_baseline.csv (averaged LGB folds)
  * XGB test  = recovered algebraically from the 2-way blend submission:
                submission_blend = 0.86*LGB + 0.14*XGB  =>  XGB = (blend - 0.86*LGB)/0.14
  * GRU test  = gru_test_pred.parquet (from gru_predict.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from metric import amex_metric_np

# weights used by the committed 2-way blend (see blend.py output)
W_LGB_2WAY = 0.86


def _oof(path):
    return pd.read_parquet(config.PROCESSED_DIR / path).set_index(config.ID_COL).sort_index()


def main() -> None:
    # ---- optimal 3-way weights on OOF --------------------------------------
    lgb, xgb, gru = _oof("oof_predictions.parquet"), _oof("oof_xgb.parquet"), _oof("oof_gru.parquet")
    assert (lgb.index == xgb.index).all() and (lgb.index == gru.index).all()
    y = lgb["target"].values
    pl, px, pg = lgb["oof_pred"].values, xgb["oof_pred"].values, gru["oof_pred"].values

    best = (None, -1.0)
    for a in np.linspace(0, 1, 41):
        for b in np.linspace(0, 1 - a, int((1 - a) * 40) + 1):
            c = 1 - a - b
            s = amex_metric_np(y, a * pl + b * px + c * pg)
            if s > best[1]:
                best = ((round(float(a), 3), round(float(b), 3), round(float(c), 3)), s)
    (wl, wx, wg), oof_score = best
    print(f"3-way OOF weights LGB={wl} XGB={wx} GRU={wg} -> amex={oof_score:.5f}")
    print(f"(vs 2-way 0.79294, best single LGB 0.79266)")

    # ---- assemble test predictions (aligned by customer_ID) -----------------
    lgb_sub = pd.read_csv(config.SUBMISSION_DIR / "submission_lgbm_baseline.csv") \
        .set_index(config.ID_COL).sort_index()
    blend_sub = pd.read_csv(config.SUBMISSION_DIR / "submission_blend.csv") \
        .set_index(config.ID_COL).sort_index()
    gru_sub = pd.read_parquet(config.PROCESSED_DIR / "gru_test_pred.parquet") \
        .set_index(config.ID_COL).sort_index()
    assert (lgb_sub.index == blend_sub.index).all() and (lgb_sub.index == gru_sub.index).all()

    t_lgb = lgb_sub["prediction"].values
    t_xgb = (blend_sub["prediction"].values - W_LGB_2WAY * t_lgb) / (1 - W_LGB_2WAY)
    t_gru = gru_sub["prediction"].values

    t_final = wl * t_lgb + wx * t_xgb + wg * t_gru
    out = config.SUBMISSION_DIR / "submission_blend3.csv"
    pd.DataFrame({config.ID_COL: lgb_sub.index, "prediction": t_final}) \
        .to_csv(out, index=False)
    print(f"wrote {out} ({len(t_final):,} rows) "
          f"min={t_final.min():.4f} mean={t_final.mean():.4f} max={t_final.max():.4f}")


if __name__ == "__main__":
    main()
