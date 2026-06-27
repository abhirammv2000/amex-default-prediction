"""Predict the test set with the trained GRU fold models (CPU is fine).

Loads seq_test.npz and the five saved gru_fold*.pt checkpoints, averages their
predictions, and writes data/processed/gru_test_pred.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import config
from train_gru import GRUClassifier, _predict, DEVICE


def main() -> None:
    td = np.load(config.PROCESSED_DIR / "seq_test.npz", allow_pickle=True)
    seq, lengths, ids = td["seq"], td["lengths"], td["customer_ids"]
    n_feat = seq.shape[2]
    idx = np.arange(len(ids))
    preds = np.zeros(len(ids), dtype=np.float32)
    for fold in range(1, config.N_FOLDS + 1):
        m = GRUClassifier(n_feat)
        m.load_state_dict(torch.load(config.MODEL_DIR / f"gru_fold{fold}.pt",
                                     map_location="cpu"))
        preds += _predict(m.to(DEVICE), seq, lengths, idx, 2048) / config.N_FOLDS
        print(f"  scored fold {fold}", flush=True)
    out = config.PROCESSED_DIR / "gru_test_pred.parquet"
    pd.DataFrame({config.ID_COL: ids, "prediction": preds}).to_parquet(out, index=False)
    print(f"wrote {out} for {len(ids):,} customers "
          f"(min={preds.min():.4f} mean={preds.mean():.4f} max={preds.max():.4f})")


if __name__ == "__main__":
    main()
