"""Trained XGBoost models -> ONNX, with a real numerical validation pass.

Separate from train.py: exporting/validating is a distinct concern from
training itself (same one-script-per-concern pattern as the rest of this
project), and this script operates on the already-saved model artifacts,
not in-memory training state.

Real finding before writing this (see proyecto-mlops-plan.md, task 6): the
standard assumption that XGBoost's native-categorical models don't export
cleanly to ONNX turned out to be wrong for this setup. Two things actually
needed, neither of them a fallback encoding:
1. onnxmltools' converter expects feature names matching the 'f%d'
   pattern, not the original column names — rename on a *copy* of the
   booster only (renaming in place breaks the original model's ability to
   `.predict()` with its real column names afterward).
2. At inference time, categorical inputs must be passed as their pandas
   category codes (int, cast to float32; unseen/NaN stays NaN — XGBoost
   routes it as missing either way), not the original string labels.
Predictions matched the original model to ~1e-5 (float32 noise), including
the unseen-category (missing) case — no accuracy tradeoff, unlike what a
fallback one-hot/ordinal re-encode would have risked.
"""

import copy
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split

from train import (
    CAT_COLS,
    FEATURE_COLS,
    MODELS_DIR,
    REPO_ROOT,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    load_training_frame,
    to_categorical,
)

TOLERANCE = 1e-3


def load_test_split(features: pd.DataFrame, operation_type: str) -> pd.DataFrame:
    """Same split mechanism as train_operation_model, including
    to_categorical fit on the train fold's own categories — needed here to
    validate against genuine held-out rows (with real unseen-category
    handling), not training rows or an independently-refit category set.
    """
    sub = features[features["operation_type"] == operation_type]
    X = sub[FEATURE_COLS]
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, X_test = to_categorical(X_train, X_test, CAT_COLS)
    return X_test


def to_onnx(model: xgb.XGBRegressor):
    """Converts a copy of the model's booster — mutating feature_names on
    the original breaks its ability to .predict() with real column names.
    """
    exportable = copy.deepcopy(model)
    exportable.get_booster().feature_names = [f"f{i}" for i in range(len(FEATURE_COLS))]
    return convert_xgboost(exportable, initial_types=[("input", FloatTensorType([None, len(FEATURE_COLS)]))])


def to_onnx_input(X: pd.DataFrame, cat_cols: list = CAT_COLS) -> np.ndarray:
    """Categorical columns -> their pandas category codes (XGBoost's own
    internal representation for categorical splits); unseen/-1 -> NaN, same
    missing-value routing as the original model.
    """
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].cat.codes.replace(-1, np.nan)
    return X[FEATURE_COLS].to_numpy(dtype=np.float32)


def validate_onnx(model: xgb.XGBRegressor, onnx_model, X_test: pd.DataFrame, tolerance: float = TOLERANCE) -> dict:
    orig_pred = model.predict(X_test)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    onnx_pred = sess.run(None, {"input": to_onnx_input(X_test)})[0].flatten()
    max_abs_diff = float(np.max(np.abs(orig_pred - onnx_pred)))
    matches = bool(np.allclose(orig_pred, onnx_pred, rtol=tolerance, atol=tolerance))
    return {"max_abs_diff": max_abs_diff, "matches": matches, "n": len(X_test)}


def save_onnx(onnx_model, operation_type: str, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / f"{operation_type.lower()}_xgb.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return path


def save_category_mapping(
    X_test: pd.DataFrame, operation_type: str, cat_cols: list = CAT_COLS, models_dir: Path = MODELS_DIR
) -> Path:
    """The Node.js inference API (task 7) has no pandas — it needs the
    exact train-fold category order to reproduce the same integer codes
    the ONNX model was validated against (category at position i -> code
    i; a value not in the list -> missing, same as an unseen category
    here). X_test's dtype already carries train's categories exactly
    (to_categorical assigns them explicitly), so no separate X_train
    reference is needed.
    """
    mapping = {c: X_test[c].cat.categories.tolist() for c in cat_cols}
    path = models_dir / f"{operation_type.lower()}_categories.json"
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    return path


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    features = load_training_frame()

    for operation_type in ["Venta", "Alquiler"]:
        model_path = MODELS_DIR / f"{operation_type.lower()}_xgb.json"
        model = xgb.XGBRegressor()
        model.load_model(model_path)

        X_test = load_test_split(features, operation_type)

        onnx_model = to_onnx(model)
        result = validate_onnx(model, onnx_model, X_test)

        path = save_onnx(onnx_model, operation_type)
        status = "MATCH" if result["matches"] else "MISMATCH"
        print(
            f"{operation_type}: {status} — max abs diff={result['max_abs_diff']:.2e} "
            f"(tolerance={TOLERANCE}) over {result['n']} test rows. Saved to "
            f"{path.relative_to(REPO_ROOT)}"
        )
        if not result["matches"]:
            raise RuntimeError(f"ONNX predictions for {operation_type} do not match the original model")

        cat_path = save_category_mapping(X_test, operation_type)
        print(f"  Category mapping saved to {cat_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
