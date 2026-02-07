import argparse
import json
import time
from pathlib import Path
from itertools import product
from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

"""`prep.py`: La entrada del script son datos `data/raw`. La salida del script son datos `data/prep`."""

logger = get_logger("prep")

KEY_COLS: list[str] = ["date_block_num", "shop_id", "item_id"]

LAG_FEATURE_COLS: list[str] = ["item_cnt_month", "item_price_mean"]
DEFAULT_LAGS: list[int] = [1, 2, 3, 6, 12]

TARGET_COL: str = "item_cnt_month"
CLIP_TARGET_MIN: int = 0
CLIP_TARGET_MAX: int = 20


"""El uso de TypedDict junto con librerías que ofrecen comportamiento tipo "MetaDict" (como la librería metadict o Pydantic)
permite estructurar diccionarios en Python con tipado estático, autocompletado de IDE y, a menudo, acceso por atributos"""


class MetaDict(TypedDict):
    first_month: int
    last_month: int
    test_month: int
    valid_month: int
    lags: list[int]
    n_rows_matrix: int
    n_features: int


def add_shop_size_category(
    tbl: pd.DataFrame, shop_col: str = "shop_id", target_col: str = "item_cnt_month"
) -> pd.DataFrame:
    """
    Clasifica las tiendas en 'small', 'average', 'large' según el total
    de ventas (target_col) usando percentiles 33% y 66%.
    """

    shop_totals = tbl.groupby(shop_col)[target_col].sum().sort_values(ascending=False)

    p33 = shop_totals.quantile(0.33)
    p66 = shop_totals.quantile(0.66)

    shop_size_map = pd.Series("average", index=shop_totals.index)
    shop_size_map.loc[shop_totals <= p33] = "small"
    shop_size_map.loc[shop_totals > p66] = "large"

    tbl_resultado = tbl.copy()
    tbl_resultado["shop_size"] = tbl_resultado[shop_col].map(shop_size_map)

    return tbl_resultado


def add_lags(
    tbl: pd.DataFrame,
    lags: Sequence[int],
    feature_cols: Sequence[str],
    key_cols: Sequence[str] = KEY_COLS,
) -> pd.DataFrame:
    """
    Agrega variables rezagadas (lag features) para `feature_cols` usando `key_cols` como llave.
    """
    tbl_resultado = tbl.copy()

    for lag in lags:
        tbl_lag = tbl_resultado[list(key_cols) + list(feature_cols)].copy()
        tbl_lag["date_block_num"] = tbl_lag["date_block_num"] + lag

        renombres = {col: f"{col}_lag_{lag}" for col in feature_cols}
        tbl_lag = tbl_lag.rename(columns=renombres)

        tbl_resultado = tbl_resultado.merge(
            tbl_lag,
            on=list(key_cols),
            how="left",
        )

    return tbl_resultado


def add_group_mean_lag(
    tbl: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    lag: int = 1,
    feature_name: str | None = None,
) -> pd.DataFrame:
    """
    Calcula la media de `target_col` por (date_block_num + group_cols),
    aplica desplazamiento temporal (+lag) y hace left join.
    """
    if feature_name is None:
        columnas_grupo_str = "_".join(group_cols)
        feature_name = f"{columnas_grupo_str}_{target_col}_mean_lag_{lag}"

    merge_cols = ["date_block_num", *group_cols]

    tbl_mean = (
        tbl.groupby(merge_cols, as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: feature_name})
    )
    tbl_mean["date_block_num"] = tbl_mean["date_block_num"] + lag

    return tbl.merge(tbl_mean, on=merge_cols, how="left")


def build_matrix(
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[str], MetaDict, pd.DataFrame]:
    logger.info("action=build_matrix status=started")

    items = pd.read_csv(raw_dir / "items.csv", encoding="utf-8", low_memory=False)
    sales_train = pd.read_csv(
        raw_dir / "sales_train.csv", encoding="utf-8", low_memory=False
    )
    test = pd.read_csv(raw_dir / "test.csv", encoding="utf-8", low_memory=False)

    logger.info(
        "action=load_data status=success "
        f"rows_items={len(items):,} rows_sales_train={len(sales_train):,} rows_test={len(test):,}"
    )

    tbl_test_pairs_with_id = test[["ID", "shop_id", "item_id"]].copy()

    tbl_sales = sales_train.copy()
    tbl_sales["date"] = pd.to_datetime(
        tbl_sales["date"], format="%d.%m.%Y", errors="coerce"
    )

    bad_dates = int(tbl_sales["date"].isna().sum())
    if bad_dates > 0:
        logger.warning(f"action=parse_dates status=warning bad_dates={bad_dates:,}")

    before_rows = len(tbl_sales)
    tbl_sales = tbl_sales[
        (tbl_sales["item_price"] > 0) & (tbl_sales["item_cnt_day"] >= 0)
    ].copy()
    after_rows = len(tbl_sales)
    if after_rows < before_rows:
        logger.info(
            "action=filter_sales status=success "
            f"rows_before={before_rows:,} rows_after={after_rows:,} dropped={before_rows - after_rows:,}"
        )

    tbl_sales["revenue_day"] = tbl_sales["item_price"] * tbl_sales["item_cnt_day"]

    first_month = int(tbl_sales["date_block_num"].min())
    last_month = int(tbl_sales["date_block_num"].max())
    test_month = last_month + 1

    logger.info(
        "action=month_range status=success "
        f"first_month={first_month} last_month={last_month} test_month={test_month}"
    )

    tbl_month_sales = tbl_sales.groupby(KEY_COLS, as_index=False).agg(
        item_cnt_month=("item_cnt_day", "sum"),
        item_price_mean=("item_price", "mean"),
        revenue_month=("revenue_day", "sum"),
    )
    tbl_month_sales = add_shop_size_category(tbl_month_sales)

    grid_arrays: list[np.ndarray] = []
    for month in range(first_month, last_month + 1):
        tbl_mes = tbl_sales[tbl_sales["date_block_num"] == month]
        shop_ids = tbl_mes["shop_id"].unique()
        item_ids = tbl_mes["item_id"].unique()
        grid_arrays.append(
            np.array(list(product([month], shop_ids, item_ids)), dtype=np.int32)
        )

    tbl_matrix = pd.DataFrame(np.vstack(grid_arrays), columns=KEY_COLS)

    tbl_test_pairs = test[["shop_id", "item_id"]].copy()
    tbl_test_pairs["date_block_num"] = test_month

    tbl_matrix = pd.concat([tbl_matrix, tbl_test_pairs[KEY_COLS]], ignore_index=True)
    tbl_matrix = tbl_matrix.merge(tbl_month_sales, on=KEY_COLS, how="left")

    shop_size_map = {"small": 0, "average": 1, "large": 2}
    tbl_matrix["shop_size"] = (
        tbl_matrix["shop_size"].fillna("average").map(shop_size_map).astype(np.int8)
    )

    for col in ["item_cnt_month", "item_price_mean", "revenue_month"]:
        tbl_matrix[col] = tbl_matrix[col].fillna(0).astype(np.float32)

    cat_map = items.set_index("item_id")["item_category_id"]
    tbl_matrix["item_category_id"] = tbl_matrix["item_id"].map(cat_map).astype(np.int16)

    tbl_matrix[TARGET_COL] = tbl_matrix[TARGET_COL].clip(
        CLIP_TARGET_MIN, CLIP_TARGET_MAX
    )

    tbl_matrix["month"] = (tbl_matrix["date_block_num"] % 12).astype(np.int8)
    tbl_matrix["year"] = (2013 + tbl_matrix["date_block_num"] // 12).astype(np.int16)

    cols_old = [c for c in tbl_matrix.columns if "lag_" in c] + [
        "shop_mean_lag_1",
        "item_mean_lag_1",
        "cat_mean_lag_1",
    ]
    tbl_matrix = tbl_matrix.drop(columns=cols_old, errors="ignore")

    lags = DEFAULT_LAGS
    tbl_matrix = add_lags(tbl_matrix, lags, feature_cols=LAG_FEATURE_COLS)

    tbl_matrix = add_group_mean_lag(
        tbl_matrix, ["shop_id"], TARGET_COL, lag=1, feature_name="shop_mean_lag_1"
    )
    tbl_matrix = add_group_mean_lag(
        tbl_matrix, ["item_id"], TARGET_COL, lag=1, feature_name="item_mean_lag_1"
    )
    tbl_matrix = add_group_mean_lag(
        tbl_matrix,
        ["item_category_id"],
        TARGET_COL,
        lag=1,
        feature_name="cat_mean_lag_1",
    )

    lag_cols = [c for c in tbl_matrix.columns if "lag_" in c] + [
        "shop_mean_lag_1",
        "item_mean_lag_1",
        "cat_mean_lag_1",
    ]
    lag_cols = list(dict.fromkeys(lag_cols))

    tbl_matrix[lag_cols] = tbl_matrix[lag_cols].fillna(0).astype(np.float32)

    feature_cols = [
        "shop_id",
        "item_id",
        "item_category_id",
        "month",
        "year",
        "shop_size",
    ] + lag_cols

    meta: MetaDict = {
        "first_month": first_month,
        "last_month": last_month,
        "test_month": test_month,
        "valid_month": last_month,
        "lags": list(lags),
        "n_rows_matrix": int(tbl_matrix.shape[0]),
        "n_features": int(len(feature_cols)),
    }

    logger.info(
        "action=build_matrix status=success "
        f"rows_matrix={tbl_matrix.shape[0]:,} n_features={len(feature_cols):,}"
    )

    return tbl_matrix, feature_cols, meta, tbl_test_pairs_with_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", default="data/raw", help="Directorio de entrada (raw)"
    )
    parser.add_argument(
        "--prep-dir", default="data/prep", help="Directorio de salida (prep)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    logger.info("action=prep status=started")

    raw_dir = Path(args.raw_dir)
    prep_dir = Path(args.prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    try:
        matrix, feature_cols, meta, test_pairs_with_id = build_matrix(raw_dir)
    except Exception as e:
        logger.error(
            "action=prep status=failure "
            f"error_type={type(e).__name__} error_message={str(e)[:200]}",
            exc_info=True,
        )
        raise

    out_matrix = prep_dir / "matrix.csv.gz"
    matrix.to_csv(out_matrix, index=False, compression="gzip")

    (prep_dir / "feature_cols.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    (prep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    test_pairs_with_id.to_csv(prep_dir / "test_pairs.csv", index=False)

    duration = time.time() - start_time
    logger.info(
        "action=prep status=success "
        f"matrix_file={out_matrix.name} feature_cols_file=feature_cols.json meta_file=meta.json "
        f"test_pairs_file=test_pairs.csv duration_seconds={duration:.2f}"
    )


if __name__ == "__main__":
    main()
