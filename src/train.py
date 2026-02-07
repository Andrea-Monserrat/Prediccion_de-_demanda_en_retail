# src/train.py

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_squared_error

from utils.logging_config import get_logger

""" `train.py`: La entrada del script son datos `data/prep`. La salida del script es un modelo entrenado"""

logger = get_logger("train")

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
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--model-file", default="model.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    logger.info("action=train status=started")

    prep_dir = Path(args.prep_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        matrix_path = prep_dir / "matrix.csv.gz"
        feature_cols_path = prep_dir / "feature_cols.json"
        meta_path = prep_dir / "meta.json"

        matrix = pd.read_csv(matrix_path, low_memory=False)
        feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        logger.info(
            "action=load_prep status=success "
            f"matrix_rows={len(matrix):,} n_features={len(feature_cols):,} "
            f"files=({matrix_path.name},{feature_cols_path.name},{meta_path.name})"
        )

        valid_month = int(meta["valid_month"])
        test_month = int(meta["test_month"])

        train_mask = matrix["date_block_num"] < valid_month
        valid_mask = matrix["date_block_num"] == valid_month

        X_train = matrix.loc[train_mask, feature_cols]
        y_train = matrix.loc[train_mask, TARGET_COL]

        X_valid = matrix.loc[valid_mask, feature_cols]
        y_valid = matrix.loc[valid_mask, TARGET_COL]

        logger.info(
            "action=split status=success "
            f"valid_month={valid_month} test_month={test_month} "
            f"train_rows={len(X_train):,} valid_rows={len(X_valid):,}"
        )

        if BASELINE_COL not in X_valid.columns:
            logger.error(
                "action=validate_inputs status=failure "
                f"missing_baseline_feature={BASELINE_COL}"
            )
            raise ValueError(
                f"Baseline feature '{BASELINE_COL}' no existe en feature_cols/prep output."
            )

        pred_baseline = X_valid[BASELINE_COL].to_numpy()
        baseline_rmse = rmse(y_valid.to_numpy(), pred_baseline)

        logger.info(
            "action=baseline status=success "
            f"baseline_feature={BASELINE_COL} baseline_rmse={baseline_rmse:.4f}"
        )

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
            logger.info(f"action=model_fit status=started model={name}")
            model.fit(X_train, y_train)
            pred = np.clip(model.predict(X_valid), CLIP_MIN, CLIP_MAX)
            score = rmse(y_valid.to_numpy(), pred)
            scores[name] = score
            fitted[name] = model
            logger.info(
                f"action=model_fit status=success model={name} rmse={score:.4f}"
            )

        best_name = min(scores, key=scores.get)
        logger.info(
            "action=model_select status=success "
            f"best_model={best_name} best_rmse={scores[best_name]:.4f}"
        )

        X_full = pd.concat([X_train, X_valid], axis=0)
        y_full = pd.concat([y_train, y_valid], axis=0)

        model_final = clone(candidates[best_name])
        model_final.fit(X_full, y_full)

        model_out = models_dir / args.model_file
        joblib.dump(model_final, model_out)

        report = {
            "baseline_rmse_lag_1": baseline_rmse,
            "scores": scores,
            "best_model": best_name,
            "valid_month": valid_month,
            "test_month": test_month,
        }

        report_path = models_dir / "train_report.json"
        cols_path = models_dir / "feature_cols.json"

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

        duration = time.time() - start_time
        logger.info(
            "action=train status=success "
            f"saved_model={model_out.name} saved_report={report_path.name} saved_feature_cols={cols_path.name} "
            f"duration_seconds={duration:.2f}"
        )

    except FileNotFoundError as e:
        logger.error(
            "action=train status=failure error_type=FileNotFoundError "
            f"error_message={str(e)[:200]}",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            "action=train status=failure "
            f"error_type={type(e).__name__} error_message={str(e)[:200]}",
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    main()
