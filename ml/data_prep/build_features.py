"""data/processed/listings.parquet -> data/processed/features.parquet.

This is what task 3's baseline model trains on directly (no Feast/Postgres
yet). Each transformation is a standalone function for the same reason as
clean_arequipa.py: reused as-is later, not re-derived.
"""

from pathlib import Path

import pandas as pd

LISTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "listings.parquet"
FEATURES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.parquet"

# Additive-smoothing constant for the district-average leave-one-out mean.
SMOOTHING_K = 10

FEATURE_COLS = ["district", "surface", "property_type", "operation_type", "district_avg_price_per_m2"]
TARGET_COL = "price_usd"


def load_listings(path: Path = LISTINGS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def smoothed_district_avg(df: pd.DataFrame, k: int = SMOOTHING_K) -> pd.Series:
    """Leave-one-out mean of price_per_m2 per (operation_type, l4), shrunk
    toward the operation_type's global mean by k pseudo-observations.

    Decision + evidence in notebooks/02_feature_eda.ipynb ("Derived feature:
    avg price/m2 by district — leakage risk"): a naive groupby mean leaks
    each row's own price into its own feature (203% self-inflation observed
    on a 6-listing district with one outlier), and plain leave-one-out is
    undefined for singleton districts. Smoothing fixes both in one formula.
    """
    price_per_m2 = df["price_usd"] / df["surface"]
    grp = price_per_m2.groupby([df["operation_type"], df["l4"]])
    count = grp.transform("count")
    total = grp.transform("sum")
    global_mean = price_per_m2.groupby(df["operation_type"]).transform("mean")
    sum_others = total - price_per_m2
    count_others = count - 1
    return (sum_others + k * global_mean) / (count_others + k)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Base features (district, surface, property/operation type) + the
    derived district_avg_price_per_m2, alongside id and the target.

    `l4` is renamed to `district` here — the feature layer gets human-
    readable names; `listings.parquet` keeps the raw dataset's l1..l6 naming.
    """
    df = df.copy()
    df["district_avg_price_per_m2"] = smoothed_district_avg(df)
    df = df.rename(columns={"l4": "district"})
    return df[["id", *FEATURE_COLS, TARGET_COL]]


def save_features(df: pd.DataFrame, path: Path = FEATURES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)


def main() -> None:
    listings = load_listings()
    print(f"Loaded {len(listings)} rows from {LISTINGS_PATH.name}")

    features = build_features(listings)
    print(f"Built features table: {features.shape[0]} rows x {features.shape[1]} cols")
    print(f"Columns: {list(features.columns)}")
    print(f"Nulls per column:\n{features.isna().sum()}")

    save_features(features)
    print(f"Saved to {FEATURES_PATH.relative_to(FEATURES_PATH.parents[2])}")


if __name__ == "__main__":
    main()
