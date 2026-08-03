"""Feast offline store -> two trained XGBoost models (Venta, Alquiler).

Every decision here (native categorical encoding, separate models per
operation_type, log(price_usd) target, random split, R2/RMSE/MAE/MAPE
metrics) is explored and justified with real evidence in
notebooks/03_baseline_model.ipynb — this script is the repeatable version.

Task 6 changes from the task 3 baseline, explored and justified in
notebooks/04_definitive_model.ipynb:

1. district/surface/property_type/operation_type are read via Feast's
   get_historical_features (point-in-time join against the offline store),
   not straight from features.parquet.
2. district_avg_price_per_m2 is dropped entirely. It was flagged since
   task 3 as leakage-prone (a single static aggregate over the whole
   dataset). Fixing it properly — fit per train fold, evaluated via 5-fold
   CV — made results *worse* and unstable (Venta test R2 as low as -1.65
   across splits) than just not having the feature at all (stable
   R2~0.43-0.59 with district/surface/property_type alone): the honest,
   non-leaky version of this feature turned out to carry no real signal
   beyond what the district categorical already provides. MODEL_PARAMS
   below was re-tuned via the same CV process against this 3-feature set.
"""

import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from feast import FeatureStore
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"
FEATURE_REPO_PATH = REPO_ROOT / "feature_repo"
MODELS_DIR = REPO_ROOT / "data" / "processed" / "models"

FEAST_FEATURE_REFS = [
    "arequipa_listings_features:district",
    "arequipa_listings_features:surface",
    "arequipa_listings_features:property_type",
    "arequipa_listings_features:operation_type",
]

CAT_COLS = ["district", "property_type"]
FEATURE_COLS = ["district", "surface", "property_type"]
TARGET_COL = "price_usd"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Re-tuned via 5-fold CV in notebooks/04_definitive_model.ipynb, against
# the 3-feature set above (task 3's max_depth=5/min_child_weight=3 was
# tuned for a feature set that included the since-dropped
# district_avg_price_per_m2 — no longer the right config once that
# feature's gone). max_depth=2/min_child_weight=20 was the most stable
# shared choice: Venta mean CV R2=0.51 (std 0.06), Alquiler mean CV
# R2=0.34 (std 0.14) — both notably more stable than shallower sweeps.
MODEL_PARAMS = dict(max_depth=2, min_child_weight=20)

MLFLOW_EXPERIMENT_NAME = "arequipa-housing-price"


def load_entity_df(path: Path = FEATURES_PATH) -> pd.DataFrame:
    """id + event_timestamp + price_usd — the entity keys and label, not
    features. Feast's FeatureView never models the label; these three
    columns aren't part of its catalog (see ml/feature_metadata.csv), so
    they come straight from the same parquet that seeded Postgres's
    `features` table, not through Feast's Python API.
    """
    df = pd.read_parquet(path)[["id", "created_on", "price_usd"]]
    df = df.rename(columns={"created_on": "event_timestamp"})
    # created_on is stored as a plain string in the parquet — Feast's SQL
    # query compares it against a real timestamp and fails outright
    # (psycopg.errors.UndefinedFunction) without this conversion.
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return df


def fetch_features_from_feast(
    entity_df: pd.DataFrame, repo_path: Path = FEATURE_REPO_PATH
) -> pd.DataFrame:
    store = FeatureStore(repo_path=str(repo_path))
    result = store.get_historical_features(
        entity_df=entity_df, features=FEAST_FEATURE_REFS
    ).to_df()
    return result.drop(columns="event_timestamp")


def load_training_frame() -> pd.DataFrame:
    """get_historical_features passes non-key, non-timestamp entity_df
    columns straight through — price_usd already comes back attached to
    the result, no separate merge needed (an earlier version of this did
    merge, and got price_usd_x/price_usd_y out of the collision).
    """
    entity_df = load_entity_df()
    return fetch_features_from_feast(entity_df)


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

    model = xgb.XGBRegressor(
        enable_categorical=True, tree_method="hist", random_state=RANDOM_STATE, **MODEL_PARAMS
    )
    model.fit(X_train, np.log(y_train))
    pred_train = np.exp(model.predict(X_train))
    pred_test = np.exp(model.predict(X_test))

    naive_pred = np.full_like(y_test, y_train.mean(), dtype=float)

    # Train metrics alongside test metrics: a large gap between them means
    # overfitting, not just a "how good is the model" number. See
    # notebooks/03_baseline_model.ipynb ("Overfitting check" and
    # "Regularization") for how big the gap was with defaults and why
    # MODEL_PARAMS above is what actually fixes the underlying stability
    # problem (single-split gap numbers are noisy, not the real signal).
    metrics = dict(
        operation_type=operation_type,
        n_train=len(X_train),
        n_test=len(X_test),
        train_r2=r2_score(y_train, pred_train),
        train_mape_pct=mean_absolute_percentage_error(y_train, pred_train) * 100,
        r2=r2_score(y_test, pred_test),
        rmse=mean_squared_error(y_test, pred_test) ** 0.5,
        mae=mean_absolute_error(y_test, pred_test),
        mape_pct=mean_absolute_percentage_error(y_test, pred_test) * 100,
        naive_mean_mape_pct=mean_absolute_percentage_error(y_test, naive_pred) * 100,
    )
    return {"model": model, "metrics": metrics}


def save_model(model: xgb.XGBRegressor, operation_type: str, models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{operation_type.lower()}_xgb.json"
    model.save_model(path)
    return path


def log_to_mlflow(operation_type: str, model: xgb.XGBRegressor, metrics: dict) -> None:
    """One MLflow run per operation_type (they're functionally distinct
    models, not versions of each other) — registers into its own named
    entry in the Model Registry, not two versions of one shared name.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=operation_type):
        mlflow.log_params({**MODEL_PARAMS, "operation_type": operation_type, "target_transform": "log(price_usd)"})
        mlflow.log_metrics(
            {
                "train_r2": metrics["train_r2"],
                "train_mape_pct": metrics["train_mape_pct"],
                "test_r2": metrics["r2"],
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
                "test_mape_pct": metrics["mape_pct"],
                "naive_mean_mape_pct": metrics["naive_mean_mape_pct"],
                "r2_gap": metrics["train_r2"] - metrics["r2"],
            }
        )
        mlflow.xgboost.log_model(
            model, name="model", registered_model_name=f"arequipa-price-{operation_type.lower()}"
        )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    features = load_training_frame()
    print(f"Loaded {len(features)} rows from Feast's offline store")

    for operation_type in ["Venta", "Alquiler"]:
        result = train_operation_model(features, operation_type)
        m = result["metrics"]
        print(
            f"\n{operation_type}: n_train={m['n_train']} n_test={m['n_test']}\n"
            f"  train: R2={m['train_r2']:.4f}  MAPE={m['train_mape_pct']:.1f}%\n"
            f"  test:  R2={m['r2']:.4f}  RMSE={m['rmse']:,.2f}  MAE={m['mae']:,.2f}  "
            f"MAPE={m['mape_pct']:.1f}%  (trivial baseline MAPE={m['naive_mean_mape_pct']:.1f}%)\n"
            f"  gap:   R2 diff={m['train_r2'] - m['r2']:.4f}   MAPE diff={m['mape_pct'] - m['train_mape_pct']:.1f}pp"
        )
        path = save_model(result["model"], operation_type)
        print(f"  Saved model to {path.relative_to(FEATURES_PATH.parents[2])}")
        log_to_mlflow(operation_type, result["model"], m)
        print(f"  Logged to MLflow experiment '{MLFLOW_EXPERIMENT_NAME}', registered as arequipa-price-{operation_type.lower()}")


if __name__ == "__main__":
    main()
