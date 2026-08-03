-- Runs once, only on a fresh data directory (docker-entrypoint-initdb.d
-- convention) — connected to POSTGRES_DB (aqp_housing) by default, hence
-- CREATE DATABASE instead of \c first.
CREATE DATABASE mlflow;
