# src/prep.py

import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


def add_lags(df: pd.DataFrame, lags: list[int], cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    key = ["date_block_num", "shop_id", "item_id"]
    for lag in lags:
        tmp = df[key + cols].copy()
        tmp["date_block_num"] = tmp["date_block_num"] + lag
        tmp = tmp.rename(columns={c: f"{c}_lag_{lag}" for c in cols})
        df = df.merge(tmp, on=key, how="left")
    return df


def add_group_mean_lag(
    df: pd.DataFrame,
    group_cols: list[str],
    col: str,
    lag: int = 1,
    name: str | None = None,
) -> pd.DataFrame:
    if name is None:
        name = "_".join(group_cols) + f"_{col}_mean_lag_{lag}"

    g = (
        df.groupby(["date_block_num"] + group_cols, as_index=False)[col]
        .mean()
        .rename(columns={col: name})
    )
    g["date_block_num"] = g["date_block_num"] + lag
    return df.merge(g, on=["date_block_num"] + group_cols, how="left")


def build_matrix(raw_dir: Path) -> tuple[pd.DataFrame, list[str], dict]:
    # ---- cargar raw (igual que notebook) ----
    item_categories = pd.read_csv(raw_dir / "item_categories.csv", encoding="utf-8", low_memory=False)
    items = pd.read_csv(raw_dir / "items.csv", encoding="utf-8", low_memory=False)
    sales_train = pd.read_csv(raw_dir / "sales_train.csv", encoding="utf-8", low_memory=False)
    test = pd.read_csv(raw_dir / "test.csv", encoding="utf-8", low_memory=False)

    # guardamos pares con ID (para submission/inferencia)
    test_pairs_with_id = test[["ID", "shop_id", "item_id"]].copy()

    # ---- limpieza + agregado mensual (igual que notebook) ----
    sales = sales_train.copy()
    sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y", errors="coerce")

    sales = sales[(sales["item_price"] > 0) & (sales["item_cnt_day"] >= 0)].copy()
    sales["revenue_day"] = sales["item_price"] * sales["item_cnt_day"]

    first_month = int(sales["date_block_num"].min())
    last_month = int(sales["date_block_num"].max())
    test_month = last_month + 1

    month_sales = (
        sales.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
        .agg(
            item_cnt_month=("item_cnt_day", "sum"),
            item_price_mean=("item_price", "mean"),
            revenue_month=("revenue_day", "sum"),
        )
    )

    # ---- grid por mes + agregar mes de test (igual que notebook) ----
    grid = []
    for m in range(first_month, last_month + 1):
        cur = sales[sales["date_block_num"] == m]
        cur_shops = cur["shop_id"].unique()
        cur_items = cur["item_id"].unique()
        grid.append(np.array(list(product([m], cur_shops, cur_items)), dtype=np.int32))

    matrix = pd.DataFrame(np.vstack(grid), columns=["date_block_num", "shop_id", "item_id"])

    test_pairs = test[["shop_id", "item_id"]].copy()
    test_pairs["date_block_num"] = test_month
    matrix = pd.concat([matrix, test_pairs[["date_block_num", "shop_id", "item_id"]]], ignore_index=True)

    matrix = matrix.merge(month_sales, on=["date_block_num", "shop_id", "item_id"], how="left")
    for c in ["item_cnt_month", "item_price_mean", "revenue_month"]:
        matrix[c] = matrix[c].fillna(0).astype(np.float32)

    # categoría desde items (igual que notebook)
    cat_map = items.set_index("item_id")["item_category_id"]
    matrix["item_category_id"] = matrix["item_id"].map(cat_map).astype(np.int16)

    # clip recomendado (igual que notebook)
    matrix["item_cnt_month"] = matrix["item_cnt_month"].clip(0, 20)

    # ---- time features (igual que notebook) ----
    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"] = (2013 + matrix["date_block_num"] // 12).astype(np.int16)

    # ---- lags (igual que notebook) ----
    cols_old = [c for c in matrix.columns if "lag_" in c] + ["shop_mean_lag_1", "item_mean_lag_1", "cat_mean_lag_1"]
    matrix = matrix.drop(columns=cols_old, errors="ignore")

    lags = [1, 2, 3, 6, 12]
    matrix = add_lags(matrix, lags, cols=["item_cnt_month", "item_price_mean"])

    matrix = add_group_mean_lag(matrix, ["shop_id"], "item_cnt_month", lag=1, name="shop_mean_lag_1")
    matrix = add_group_mean_lag(matrix, ["item_id"], "item_cnt_month", lag=1, name="item_mean_lag_1")
    matrix = add_group_mean_lag(matrix, ["item_category_id"], "item_cnt_month", lag=1, name="cat_mean_lag_1")

    lag_cols = [c for c in matrix.columns if "lag_" in c] + ["shop_mean_lag_1", "item_mean_lag_1", "cat_mean_lag_1"]
    lag_cols = list(dict.fromkeys(lag_cols))  # dedup conservando orden

    matrix[lag_cols] = matrix[lag_cols].fillna(0).astype(np.float32)

    feature_cols = ["shop_id", "item_id", "item_category_id", "month", "year"] + lag_cols

    meta = {
        "first_month": first_month,
        "last_month": last_month,
        "test_month": test_month,
        "valid_month": last_month,
        "lags": lags,
        "n_rows_matrix": int(matrix.shape[0]),
        "n_features": int(len(feature_cols)),
    }

    return matrix, feature_cols, meta, test_pairs_with_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Directorio de entrada (raw)")
    parser.add_argument("--prep-dir", default="data/prep", help="Directorio de salida (prep)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    prep_dir = Path(args.prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    matrix, feature_cols, meta, test_pairs_with_id = build_matrix(raw_dir)

    # outputs
    out_matrix = prep_dir / "matrix.csv.gz"
    matrix.to_csv(out_matrix, index=False, compression="gzip")

    (prep_dir / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (prep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    test_pairs_with_id.to_csv(prep_dir / "test_pairs.csv", index=False)

    print("OK prep")
    print("matrix:", out_matrix)
    print("feature_cols:", prep_dir / "feature_cols.json")
    print("meta:", prep_dir / "meta.json")
    print("test_pairs:", prep_dir / "test_pairs.csv")


if __name__ == "__main__":
    main()
