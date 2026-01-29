# src/train.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, PoissonRegressor


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep-dir", default="data/prep")
    p.add_argument("--models-dir", default="artifacts")
    p.add_argument("--model-file", default="model.joblib")
    args = p.parse_args()

    prep_dir = Path(args.prep_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(prep_dir / "matrix.csv.gz", low_memory=False)
    feature_cols = json.loads((prep_dir / "feature_cols.json").read_text(encoding="utf-8"))
    meta = json.loads((prep_dir / "meta.json").read_text(encoding="utf-8"))

    valid_month = int(meta["valid_month"])
    test_month = int(meta["test_month"])

    train_mask = matrix["date_block_num"] < valid_month
    valid_mask = matrix["date_block_num"] == valid_month
    _ = matrix["date_block_num"] == test_month  # por consistencia con el notebook

    X_train = matrix.loc[train_mask, feature_cols]
    y_train = matrix.loc[train_mask, "item_cnt_month"]

    X_valid = matrix.loc[valid_mask, feature_cols]
    y_valid = matrix.loc[valid_mask, "item_cnt_month"]

    pred_bl = X_valid["item_cnt_month_lag_1"].values
    baseline_rmse = rmse(y_valid, pred_bl)

    model_hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=8,
        learning_rate=0.08,
        max_iter=400,
        random_state=42,
    )
    model_ridge = Ridge(alpha=1.0)
    model_pois = PoissonRegressor(alpha=1e-4, max_iter=200)

    candidates = {
        "HistGradientBoosting": model_hgb,
        "Ridge": model_ridge,
        "PoissonRegressor": model_pois,
    }

    scores = {}
    fitted = {}

    for name, m in candidates.items():
        m.fit(X_train, y_train)
        pred = np.clip(m.predict(X_valid), 0, 20)
        scores[name] = rmse(y_valid, pred)
        fitted[name] = m

    best_name = min(scores, key=scores.get)

    X_full = pd.concat([X_train, X_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)

    model_final = clone(fitted[best_name])
    model_final.fit(X_full, y_full)

    joblib.dump(model_final, models_dir / args.model_file)

    (models_dir / "train_report.json").write_text(
        json.dumps(
            {
                "baseline_rmse_lag_1": baseline_rmse,
                "scores": scores,
                "best_model": best_name,
                "valid_month": valid_month,
                "test_month": test_month,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (models_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )

    print("OK train")
    print("Baseline RMSE (lag_1):", round(baseline_rmse, 4))
    print("Scores:", {k: round(v, 4) for k, v in scores.items()})
    print("Best:", best_name)
    print("Saved:", models_dir / args.model_file)


if __name__ == "__main__":
    main()
