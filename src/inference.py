# src/inference.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from prep import build_matrix 

"""- `inference.py`: La entrada de este script son datos `data/inference` y el modelo entrenado `model.joblib`. 
La salida de este script son predicciones en batch que se guardan en `data/predictions`."""


TARGET_COL = "item_cnt_month"
CLIP_MIN = 0
CLIP_MAX = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", default="data/inference")
    parser.add_argument("--models-dir", default="artifacts")
    parser.add_argument("--pred-dir", default="data/predictions")
    parser.add_argument("--model-file", default="model.joblib")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    inference_dir = Path(args.inference_dir)
    models_dir = Path(args.models_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(models_dir / args.model_file)
    feature_cols = json.loads((models_dir / "feature_cols.json").read_text(encoding="utf-8"))

    matrix, _, meta, _ = build_matrix(inference_dir)
    test_month = int(meta["test_month"])

    test_raw = pd.read_csv(inference_dir / "test.csv", encoding="utf-8", low_memory=False)

    test_rows = matrix.loc[matrix["date_block_num"] == test_month].copy()

    missing = [c for c in feature_cols if c not in test_rows.columns]
    if missing:
        raise ValueError(f"Faltan columnas en matrix para inferencia: {missing[:10]} (total={len(missing)})")

    X_test = test_rows[feature_cols]

    # Asegura orden exacto de columnas si el modelo lo guarda
    if hasattr(model, "feature_names_in_"):
        X_test = X_test.reindex(columns=list(model.feature_names_in_))

    preds = np.clip(model.predict(X_test), CLIP_MIN, CLIP_MAX)

    pred_tbl = test_rows[["shop_id", "item_id"]].copy()
    pred_tbl[TARGET_COL] = preds

    # Merge para mapear a test.csv
    out = test_raw.merge(pred_tbl, on=["shop_id", "item_id"], how="left")

    out_path = pred_dir / "predictions.csv"
    out.to_csv(out_path, index=False)

    print("OK inference")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
