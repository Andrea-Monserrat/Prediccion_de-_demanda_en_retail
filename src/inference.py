# src/inference.py

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prep import build_matrix
from utils.logging_config import get_logger

"""- `inference.py`: La entrada de este script son datos `data/inference` y el modelo entrenado `model.joblib`.
La salida de este script son predicciones en batch que se guardan en `data/predictions`."""

logger = get_logger("inference")

TARGET_COL = "item_cnt_month"
CLIP_MIN = 0
CLIP_MAX = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-dir", default="data/inference")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--pred-dir", default="data/predictions")
    parser.add_argument("--model-file", default="model.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    logger.info("action=inference status=started")

    inference_dir = Path(args.inference_dir)
    models_dir = Path(args.models_dir)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / args.model_file
    feature_cols_path = models_dir / "feature_cols.json"
    test_path = inference_dir / "test.csv"

    try:
        if not model_path.exists():
            logger.error(
                "action=validate_inputs status=failure missing_file=%s", model_path.name
            )
            raise FileNotFoundError(model_path)

        if not feature_cols_path.exists():
            logger.error(
                "action=validate_inputs status=failure missing_file=%s",
                feature_cols_path.name,
            )
            raise FileNotFoundError(feature_cols_path)

        if not test_path.exists():
            logger.error(
                "action=validate_inputs status=failure missing_file=%s", test_path.name
            )
            raise FileNotFoundError(test_path)

        model = joblib.load(model_path)
        feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))
        logger.info(
            "action=load_model status=success model_file=%s n_features=%s",
            model_path.name,
            len(feature_cols),
        )

        matrix, _, meta, _ = build_matrix(inference_dir)
        test_month = int(meta["test_month"])
        logger.info(
            "action=build_matrix status=success rows_matrix=%s test_month=%s",
            f"{matrix.shape[0]:,}",
            test_month,
        )

        test_raw = pd.read_csv(test_path, encoding="utf-8", low_memory=False)
        logger.info(
            "action=load_test status=success test_rows=%s", f"{len(test_raw):,}"
        )

        test_rows = matrix.loc[matrix["date_block_num"] == test_month].copy()
        logger.info(
            "action=select_test_rows status=success rows=%s", f"{len(test_rows):,}"
        )

        missing = [c for c in feature_cols if c not in test_rows.columns]
        if missing:
            logger.error(
                "action=validate_features status=failure missing_cols=%s missing_total=%s",
                missing[:10],
                len(missing),
            )
            raise ValueError(
                f"Faltan columnas en matrix para inferencia: {missing[:10]} (total={len(missing)})"
            )

        X_test = test_rows[feature_cols]
        nan_cells = int(X_test.isna().sum().sum())
        if nan_cells > 0:
            logger.warning(
                "action=validate_features status=warning nan_cells=%s", f"{nan_cells:,}"
            )

        # Asegura orden exacto de columnas si el modelo lo guarda
        if hasattr(model, "feature_names_in_"):
            X_test = X_test.reindex(columns=list(model.feature_names_in_))
            logger.info("action=reindex_features status=success")

        preds = np.clip(model.predict(X_test), CLIP_MIN, CLIP_MAX)
        logger.info("action=predict status=success n_predictions=%s", f"{len(preds):,}")

        pred_tbl = test_rows[["shop_id", "item_id"]].copy()
        pred_tbl[TARGET_COL] = preds

        out = test_raw.merge(pred_tbl, on=["shop_id", "item_id"], how="left")

        out_path = pred_dir / "predictions.csv"
        out.to_csv(out_path, index=False)

        duration = time.time() - start_time
        logger.info(
            "action=inference status=success output_file=%s rows_out=%s duration_seconds=%.2f",
            out_path.name,
            f"{len(out):,}",
            duration,
        )

        print("OK inference")
        print("Saved:", out_path)

    except Exception as e:
        logger.error(
            "action=inference status=failure error_type=%s error_message=%s",
            type(e).__name__,
            str(e)[:200],
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    main()
