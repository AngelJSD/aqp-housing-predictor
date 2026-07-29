"""data/processed/features.parquet -> two trained XGBoost models (Venta, Alquiler).

Every decision here (native categorical encoding, separate models per
operation_type, log(price_usd) target, random split, R2/RMSE/MAE/MAPE
metrics) is explored and justified with real evidence in
notebooks/03_baseline_model.ipynb — this script is the repeatable version.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"

CAT_COLS = ["district", "property_type"]
FEATURE_COLS = ["district", "surface", "property_type", "district_avg_price_per_m2"]
TARGET_COL = "price_usd"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def to_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list) -> tuple:
    """Cast to pandas category dtype using train's categories. Values unseen
    in train (e.g. a district with too few listings to land in the split)
    become NaN, which XGBoost routes natively as missing — same handling
    needed at inference time for genuinely new districts.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cols:
        train_df[c] = train_df[c].astype("category")
        categories = train_df[c].cat.categories
        test_df[c] = test_df[c].where(test_df[c].isin(categories))
        test_df[c] = test_df[c].astype(pd.CategoricalDtype(categories=categories))
    return train_df, test_df


def train_operation_model(df: pd.DataFrame, operation_type: str) -> dict:
    """Train + evaluate one XGBoost model for a single operation_type.

    Decisions behind every choice below are in notebooks/03_baseline_model.ipynb:
    native categorical encoding beats one-hot, one model per operation_type
    beats a unified model (catastrophically so on Alquiler with a raw price
    target), log(price_usd) beats raw price for both models, and a plain
    random split is used since district-stratified splitting is mechanically
    impossible (some districts have exactly 1 listing).
    """
    sub = df[df["operation_type"] == operation_type]
    X = sub[FEATURE_COLS]
    y = sub[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_test = to_categorical(X_train, X_test, CAT_COLS)

    model = xgb.XGBRegressor(enable_categorical=True, tree_method="hist", random_state=RANDOM_STATE)
    model.fit(X_train, np.log(y_train))
    pred = np.exp(model.predict(X_test))

    naive_pred = np.full_like(y_test, y_train.mean(), dtype=float)

    metrics = dict(
        operation_type=operation_type,
        n_train=len(X_train),
        n_test=len(X_test),
        r2=r2_score(y_test, pred),
        rmse=mean_squared_error(y_test, pred) ** 0.5,
        mae=mean_absolute_error(y_test, pred),
        mape_pct=mean_absolute_percentage_error(y_test, pred) * 100,
        naive_mean_mape_pct=mean_absolute_percentage_error(y_test, naive_pred) * 100,
    )
    return {"model": model, "metrics": metrics}


def save_model(model: xgb.XGBRegressor, operation_type: str, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{operation_type.lower()}_xgb.json"
    model.save_model(path)
    return path


def main() -> None:
    features = load_features()
    print(f"Loaded {len(features)} rows from {FEATURES_PATH.name}")

    for operation_type in ["Venta", "Alquiler"]:
        result = train_operation_model(features, operation_type)
        m = result["metrics"]
        print(
            f"\n{operation_type}: n_train={m['n_train']} n_test={m['n_test']}\n"
            f"  R2={m['r2']:.4f}  RMSE={m['rmse']:,.2f}  MAE={m['mae']:,.2f}  "
            f"MAPE={m['mape_pct']:.1f}%  (trivial baseline MAPE={m['naive_mean_mape_pct']:.1f}%)"
        )
        path = save_model(result["model"], operation_type)
        print(f"  Saved model to {path.relative_to(FEATURES_PATH.parents[2])}")


if __name__ == "__main__":
    main()
