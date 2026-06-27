"""5-fold GRU over the raw monthly statement sequences.

A bidirectional GRU reads each customer's padded ``[13, F]`` sequence (packed so
padding is ignored) and predicts default from the final hidden state. It uses the
**same StratifiedKFold split** as the LightGBM / XGBoost models, so its OOF
predictions align row-for-row and can be blended. Early stopping is on the
official Amex metric.

This is an *exploration of temporal signal + ensemble diversity*, not the
production model — GBDTs remain the model of record (see README).

Outputs: data/processed/oof_gru.parquet, outputs/models/gru_fold*.pt,
outputs/models/cv_metadata_gru.json; if seq_test.npz exists, a test prediction
file data/processed/gru_test_pred.parquet.

Usage:
    python train_gru.py --epochs 20 --batch 1024 --hidden 128
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import config
from metric import amex_metric_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GRUClassifier(nn.Module):
    def __init__(self, n_feat, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                  # h: [layers*2, B, hidden]
        last = torch.cat([h[-2], h[-1]], dim=1)  # both directions of last layer
        return self.head(last).squeeze(1)


def _loader(seq, lengths, y, idx, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(seq[idx]).float(),
                       torch.from_numpy(lengths[idx]).long(),
                       torch.from_numpy(y[idx]).float() if y is not None
                       else torch.zeros(len(idx)))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=(DEVICE == "cuda"))


def _predict(model, seq, lengths, idx, batch):
    model.eval()
    out = np.zeros(len(idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sl = idx[s:s + batch]
            x = torch.from_numpy(seq[sl]).float().to(DEVICE)
            ln = torch.from_numpy(lengths[sl]).long().to(DEVICE)
            with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                out[s:s + batch] = torch.sigmoid(model(x, ln)).float().cpu().numpy()
    return out


def main(args) -> None:
    t0 = time.time()
    print(f"device = {DEVICE}")
    data = np.load(config.PROCESSED_DIR / "seq_train.npz", allow_pickle=True)
    seq, lengths, y = data["seq"], data["lengths"], data["labels"].astype(np.float32)
    ids = data["customer_ids"]
    n_feat = seq.shape[2]
    print(f"sequences {seq.shape} | default rate {y.mean():.4f}")

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_scores = []

    for fold, (tr, va) in enumerate(skf.split(seq[:, 0, 0], y), 1):
        model = GRUClassifier(n_feat, args.hidden, args.layers, args.dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))
        lossfn = nn.BCEWithLogitsLoss()
        tr_loader = _loader(seq, lengths, y, tr, args.batch, True)

        best, best_state, patience = -1.0, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for x, ln, yb in tr_loader:
                x, ln, yb = x.to(DEVICE), ln.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                    loss = lossfn(model(x, ln), yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            val_pred = _predict(model, seq, lengths, va, args.batch)
            score = amex_metric_np(y[va], val_pred)
            if score > best:
                best, best_state, patience = score, {k: v.cpu().clone()
                                                     for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
            print(f"  [fold {fold} ep {epoch:2d}] amex={score:.5f} best={best:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if patience >= args.patience:
                break

        model.load_state_dict(best_state)
        oof[va] = _predict(model, seq, lengths, va, args.batch)
        fold_scores.append(amex_metric_np(y[va], oof[va]))
        torch.save(best_state, config.MODEL_DIR / f"gru_fold{fold}.pt")
        print(f"[fold {fold}] amex={fold_scores[-1]:.5f}", flush=True)

    cv = amex_metric_np(y, oof)
    print(f"\n=== GRU CV ===  per-fold {[round(s,5) for s in fold_scores]}")
    print(f"OOF amex = {cv:.5f}  (LGB v2 = 0.79266)")

    pd.DataFrame({config.ID_COL: ids, "target": y.astype(int), "oof_pred": oof}) \
        .to_parquet(config.PROCESSED_DIR / "oof_gru.parquet", index=False)
    (config.MODEL_DIR / "cv_metadata_gru.json").write_text(json.dumps({
        "model": "gru", "cv_oof_amex": float(cv),
        "fold_scores": [float(s) for s in fold_scores],
        "n_features": int(n_feat), "params": vars(args)}, indent=2))

    # --- optional test inference --------------------------------------------
    test_path = config.PROCESSED_DIR / "seq_test.npz"
    if test_path.exists():
        td = np.load(test_path, allow_pickle=True)
        tseq, tlen, tids = td["seq"], td["lengths"], td["customer_ids"]
        preds = np.zeros(len(tids), dtype=np.float32)
        for fold in range(1, config.N_FOLDS + 1):
            m = GRUClassifier(n_feat, args.hidden, args.layers, args.dropout).to(DEVICE)
            m.load_state_dict(torch.load(config.MODEL_DIR / f"gru_fold{fold}.pt"))
            preds += _predict(m, tseq, tlen, np.arange(len(tids)), args.batch) / config.N_FOLDS
        pd.DataFrame({config.ID_COL: tids, "prediction": preds}) \
            .to_parquet(config.PROCESSED_DIR / "gru_test_pred.parquet", index=False)
        print(f"wrote test predictions for {len(tids):,} customers")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    main(ap.parse_args())
