# Official image ships without a Postgres driver (confirmed: `pip show
# psycopg2` finds nothing) — needed since MLFLOW_BACKEND_STORE_URI points
# at Postgres, not the sqlite/file default.
FROM ghcr.io/mlflow/mlflow:v3.12.0
RUN pip install --no-cache-dir psycopg2-binary>=2.9
