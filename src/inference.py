# src/inference.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from prep import build_matrix   # si lo sigues corriendo como: python src/inference.py


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inference-dir", default="data/inference")
    p.add_argument("--models-dir", default="artifacts")
    p.add_argument("--pred-dir", default="data/predictions")
    p.add_argument("--model-file", default="model.joblib")
    args = p.parse_args()

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
    X_test = test_rows[feature_cols]

    if hasattr(model, "feature_names_in_"):
        X_test = X_test.reindex(columns=list(model.feature_names_in_))

    preds = np.clip(model.predict(X_test), 0, 20)

    pred_df = test_rows[["shop_id", "item_id"]].copy()
    pred_df["item_cnt_month"] = preds

    out = test_raw.merge(pred_df, on=["shop_id", "item_id"], how="left")

    out_path = pred_dir / "predictions.csv"
    out.to_csv(out_path, index=False)

    print("OK inference")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
