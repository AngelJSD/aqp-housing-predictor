"""ml/feature_metadata.csv -> Postgres `feature_metadata` table.

Separate from the Postgres init script (which only creates the empty
table schema, once, on a fresh volume) — this loader upserts by `name` so
it can be re-run any time the CSV changes, without touching the volume.
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from db import get_connection

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "ml" / "feature_metadata.csv"

COLUMNS = [
    "name", "description", "dtype", "source_columns",
    "transformation", "feast_feature_view", "owner", "created_at", "version",
]


def load_csv(path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.date
    return df


def upsert_feature_metadata(conn, df: pd.DataFrame) -> None:
    rows = list(df[COLUMNS].itertuples(index=False, name=None))
    update_cols = [c for c in COLUMNS if c != "name"]
    query = f"""
        INSERT INTO feature_metadata ({", ".join(COLUMNS)})
        VALUES %s
        ON CONFLICT (name) DO UPDATE SET
            {", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)}
    """
    with conn.cursor() as cur:
        execute_values(cur, query, rows)
    conn.commit()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    df = load_csv()
    conn = get_connection()
    try:
        upsert_feature_metadata(conn, df)
    finally:
        conn.close()
    print(f"Upserted {len(df)} rows into feature_metadata from {CSV_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
