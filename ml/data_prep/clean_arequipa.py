"""Raw pe_properties.csv -> clean Arequipa `listings` table.

Each cleaning rule is a standalone function so task 10 (real current
listings, injected as "production traffic") can reuse the same rules
instead of duplicating them — see proyecto-mlops-plan.md, task 1.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "pe_properties.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "listings.parquet"

# BCRP 2020 annual average was S/3.494 per USD; rounded to a clean fixed
# constant per the plan's "tipo de cambio fijo razonable para 2020" scope.
PEN_PER_USD = 3.5

# Columns that define "same listing" for dedup purposes: everything except
# the row id and the posting dates, which naturally differ on a repost.
CONTENT_COLS = [
    "lat", "lon", "l1", "l2", "l3", "l4", "l5",
    "rooms", "bedrooms", "bathrooms",
    "surface_total", "surface_covered",
    "price", "currency", "price_period",
    "title", "description", "property_type", "operation_type",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def filter_arequipa(df: pd.DataFrame) -> pd.DataFrame:
    """l2 is the department-level column (25 unique values, 0 nulls); l4 is district."""
    return df[df["l2"] == "Arequipa"].copy()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop reposts: same listing content under a different id/posting date.

    Keeps the earliest occurrence (by created_on) of each duplicate group.
    """
    return (
        df.sort_values("created_on")
        .drop_duplicates(subset=CONTENT_COLS, keep="first")
        .sort_index()
    )


def coalesce_surface(df: pd.DataFrame) -> pd.DataFrame:
    """Add `surface` = surface_total, falling back to surface_covered when null.

    Decision + reasoning in notebooks/01_eda_arequipa.ipynb ("Decision: how to
    define superficie"): the two columns agree (median diff 0) when both are
    present, and surface is only moderately predictive of price, so this is a
    strict-superset rescue of rows with no reason not to take it.
    """
    df = df.copy()
    df["surface"] = df["surface_total"].fillna(df["surface_covered"])
    return df


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing price, surface (coalesced), or district (l4) — not imputable without inventing data."""
    return df.dropna(subset=["price", "surface", "l4"])


def normalize_currency(df: pd.DataFrame, pen_per_usd: float = PEN_PER_USD) -> pd.DataFrame:
    """Convert price to a single unit (USD) using a fixed 2020 rate.

    USD chosen as the target since it's already the majority currency in the
    data and the convention for real estate pricing in Peru. Rows with
    unknown currency are dropped — not safely convertible/imputable.
    """
    df = df[df["currency"].notna()].copy()
    rate = {"USD": 1.0, "PEN": 1.0 / pen_per_usd}
    df["price_usd"] = df["price"] * df["currency"].map(rate)
    return df


def filter_price_outliers(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Drop rows outside the [1%, 99%] price_usd range, computed per operation_type.

    Venta and Alquiler prices are on completely different scales (median
    sale price is ~150x median rent) — a global percentile cut would barely
    touch Venta's tail while wrongly gutting cheap-but-legitimate Alquiler
    listings. Percentiles must be computed within each group.
    """
    lo = df.groupby("operation_type")["price_usd"].transform("quantile", lower)
    hi = df.groupby("operation_type")["price_usd"].transform("quantile", upper)
    return df[(df["price_usd"] >= lo) & (df["price_usd"] <= hi)]


def filter_surface_sanity(df: pd.DataFrame, min_surface: float = 10.0) -> pd.DataFrame:
    """Drop rows with implausibly small surface — data-entry errors, not
    real listings.

    Found via task 3's Out-of-Time validation (external ML-engineer review):
    filter_price_outliers only filters price, never price/m2, so rows like
    surface=2 / price_usd=$2.148M (implied $1,074,000/m2) survived and
    distorted district_avg_price_per_m2 enough to break OoT stability.
    A percentile cut on price/m2 was considered and rejected — it would drop
    ~130 rows, mostly legitimate (e.g. large cheap rural lots have a real,
    low price/m2). An absolute floor is more surgical: every row below 10 m2
    checked was an obvious missing-digit typo (a "Local comercial" at
    surface=4 priced like it's 40, etc.) — no property_type in this dataset
    legitimately lists under 10 m2.
    """
    return df[df["surface"] >= min_surface]


def impute_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing lat/lon with the mean centroid of their district (l4).

    No-op on this dataset in practice: the raw ~3.8% lat/lon null rate fully
    overlaps with rows already dropped by drop_incomplete/normalize_currency/
    filter_price_outliers, so 0 rows reach this step still missing geo. Kept
    as a real transform (not just an assertion) because it's reused for task
    10's manually-collected batch, which is a different source and may not
    have that same overlap.
    """
    df = df.copy()
    df["lat"] = df["lat"].fillna(df.groupby("l4")["lat"].transform("mean"))
    df["lon"] = df["lon"].fillna(df.groupby("l4")["lon"].transform("mean"))
    return df


def save_listings(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Write the clean `listings` table. Parquet keeps dtypes (the raw CSV's
    type ambiguity — e.g. null vs empty string — was part of why the EDA
    step mattered) and is what the downstream feature-definition step (task
    2) and baseline training (task 3) read from directly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False)


def main() -> None:
    df = load_raw()
    aqp = filter_arequipa(df)
    print(f"Arequipa subset: {len(aqp)} rows")

    deduped = deduplicate(aqp)
    print(f"After dedup: {len(deduped)} rows ({len(aqp) - len(deduped)} reposts dropped)")

    complete = drop_incomplete(coalesce_surface(deduped))
    print(f"After dropping incomplete rows: {len(complete)} rows ({len(deduped) - len(complete)} dropped)")

    normalized = normalize_currency(complete)
    print(f"After currency normalization: {len(normalized)} rows ({len(complete) - len(normalized)} dropped, unknown currency)")

    no_outliers = filter_price_outliers(normalized)
    print(f"After price outlier filtering: {len(no_outliers)} rows ({len(normalized) - len(no_outliers)} dropped)")

    sane_surface = filter_surface_sanity(no_outliers)
    print(f"After surface sanity filter: {len(sane_surface)} rows ({len(no_outliers) - len(sane_surface)} dropped)")

    missing_geo_before = int((sane_surface["lat"].isna() | sane_surface["lon"].isna()).sum())
    geo_complete = impute_geo(sane_surface)
    missing_geo_after = int((geo_complete["lat"].isna() | geo_complete["lon"].isna()).sum())
    print(f"Geo imputation: {missing_geo_before} rows missing lat/lon before, {missing_geo_after} after")

    save_listings(geo_complete)
    print(f"Saved {len(geo_complete)} rows to {PROCESSED_PATH.relative_to(PROCESSED_PATH.parents[2])}")


if __name__ == "__main__":
    main()
