-- Schema only, same split as feature_metadata: this is Feast's offline
-- store source table (task 5) — values get (re)loaded by a separate
-- script any time data/processed/features.parquet regenerates, without
-- touching the volume.
CREATE TABLE features (
    id                          TEXT PRIMARY KEY,
    created_on                  TIMESTAMP NOT NULL,
    district                    TEXT NOT NULL,
    surface                     DOUBLE PRECISION NOT NULL,
    property_type               TEXT NOT NULL,
    operation_type              TEXT NOT NULL,
    district_avg_price_per_m2   DOUBLE PRECISION NOT NULL,
    price_usd                   DOUBLE PRECISION NOT NULL
);
