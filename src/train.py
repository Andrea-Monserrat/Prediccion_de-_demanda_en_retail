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

""" `train.py`: La entrada del script son datos `data/prep`. La salida del script es un modelo entrenado"""


TARGET_COL = "item_cnt_month"
BASELINE_COL = "item_cnt_month_lag_1"
CLIP_MIN = 0
CLIP_MAX = 20
RANDOM_SEED = 42

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep-dir", default="data/prep")
    parser.add_argument("--models-dir", default="artifacts")
    parser.add_argument("--model-file", default="model.joblib")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

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

    X_train = matrix.loc[train_mask, feature_cols]
    y_train = matrix.loc[train_mask, TARGET_COL]

    X_valid = matrix.loc[valid_mask, feature_cols]
    y_valid = matrix.loc[valid_mask, TARGET_COL]

    if BASELINE_COL not in X_valid.columns:
        raise ValueError(f"Baseline feature '{BASELINE_COL}' no existe en feature_cols/prep output.")

    pred_baseline = X_valid[BASELINE_COL].to_numpy()
    baseline_rmse = rmse(y_valid.to_numpy(), pred_baseline)

    candidates = {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=8,
            learning_rate=0.08,
            max_iter=400,
            random_state=RANDOM_SEED,
        ),
        "Ridge": Ridge(alpha=1.0),
        "PoissonRegressor": PoissonRegressor(alpha=1e-4, max_iter=200),
    }

    scores: dict[str, float] = {}
    fitted: dict[str, object] = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_valid), CLIP_MIN, CLIP_MAX)
        scores[name] = rmse(y_valid.to_numpy(), pred)
        fitted[name] = model

    best_name = min(scores, key=scores.get)

    X_full = pd.concat([X_train, X_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)

    # Re-entrenar desde el estimador base (sin estado) para evitar problemas de clonar un fitted model
    model_final = clone(candidates[best_name])
    model_final.fit(X_full, y_full)

    joblib.dump(model_final, models_dir / args.model_file)

    report = {
        "baseline_rmse_lag_1": baseline_rmse,
        "scores": scores,
        "best_model": best_name,
        "valid_month": valid_month,
        "test_month": test_month,
    }

    (models_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (models_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    print("OK train")
    print("Baseline RMSE (lag_1):", round(baseline_rmse, 4))
    print("Scores:", {k: round(v, 4) for k, v in scores.items()})
    print("Best:", best_name)
    print("Saved:", models_dir / args.model_file)


if __name__ == "__main__":
    main()
