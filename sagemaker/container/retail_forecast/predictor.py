from __future__ import annotations

import io
import json
import os

import flask
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "/opt/ml/model"

CLIP_MIN = 0
CLIP_MAX = 20


class ScoringService:
    model = None
    feature_cols = None
    test_features = None

    @classmethod
    def cargar(cls) -> bool:
        if cls.model is not None:
            return True
        try:
            cls.model = joblib.load(os.path.join(MODEL_PATH, "model.joblib"))

            with open(os.path.join(MODEL_PATH, "feature_cols.json")) as f:
                cls.feature_cols = json.load(f)

            df = pd.read_csv(
                os.path.join(MODEL_PATH, "test_features.csv.gz"),
                compression="gzip",
                low_memory=False,
            )
            cls.test_features = df.set_index(["shop_id", "item_id"])

            print(
                f"action=cargar_modelo status=ok "
                f"n_features={len(cls.feature_cols)} "
                f"test_rows={len(cls.test_features)}"
            )
            return True
        except Exception as exc:
            print(f"action=cargar_modelo status=error error={exc}")
            return False

    @classmethod
    def predecir(cls, pares: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(pares[["shop_id", "item_id"]])
        features = cls.test_features.reindex(idx)[cls.feature_cols].fillna(0)
        return np.clip(cls.model.predict(features), CLIP_MIN, CLIP_MAX)


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    status = 200 if ScoringService.cargar() else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    if flask.request.content_type != "text/csv":
        return flask.Response(
            response="Solo se acepta Content-Type: text/csv",
            status=415,
            mimetype="text/plain",
        )

    if not ScoringService.cargar():
        return flask.Response(
            response="Modelo no disponible",
            status=503,
            mimetype="text/plain",
        )

    pares = pd.read_csv(io.StringIO(flask.request.data.decode("utf-8")))

    if "shop_id" not in pares.columns or "item_id" not in pares.columns:
        return flask.Response(
            response="El CSV debe tener columnas: shop_id, item_id",
            status=400,
            mimetype="text/plain",
        )

    print(f"action=invocations n_pares={len(pares)}")

    predicciones = ScoringService.predecir(pares)

    resultado = pares[["shop_id", "item_id"]].copy()
    resultado["item_cnt_month"] = predicciones

    out = io.StringIO()
    resultado.to_csv(out, index=False)

    return flask.Response(response=out.getvalue(), status=200, mimetype="text/csv")
