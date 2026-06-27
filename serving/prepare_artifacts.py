"""Export the artifacts the serving pipeline needs from the training outputs:
the canonical feature-column order and the train-fit categorical maps.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)

names = [c for c in pq.read_schema(config.TRAIN_FEATURES).names if c != config.ID_COL]
(ART / "feature_names.json").write_text(json.dumps(names))
shutil.copy(config.PROCESSED_DIR / "categorical_maps.json", ART / "categorical_maps.json")
print(f"saved {len(names)} feature names + categorical_maps.json to {ART}")
