-- Schema only — no rows. Loading ml/feature_metadata.csv is a separate
-- script (run after `docker compose up`), not part of container init, so
-- the catalog can be reloaded without wiping the Postgres volume.
CREATE TABLE feature_metadata (
    name                TEXT PRIMARY KEY,
    description         TEXT NOT NULL,
    dtype               TEXT NOT NULL,
    source_columns      TEXT NOT NULL,
    transformation      TEXT NOT NULL,
    feast_feature_view  TEXT NOT NULL,
    owner               TEXT NOT NULL,
    created_at          DATE NOT NULL,
    version             INTEGER NOT NULL
);
