"""Postgres `features` (training) vs `prediction_logs` (recent traffic)
-> Evidently HTML drift reports, one per operation_type.

This is input-feature drift (district/surface/property_type), not
prediction-accuracy drift — there's no ground-truth price for served
requests to compare against. Real, interesting drift is task 10's job
(a manually collected batch of current listings); running this now just
validates the mechanism against whatever's in prediction_logs so far
(task 7's smoke-test calls, same 2020-like distribution) — not meant to
show a dramatic drift result yet.

NLTK_DISABLE_IMPORT_SECURITY must be set before importing evidently: this
project's venv lives inside the repo, so NLTK's CWE-427 import-hijacking
guard (blocks imports resolving under the CWD) false-positives on its own
`regex` dependency — confirmed with `inspect`, not guessed. Real venvs
nested under the project root are the common case, not the attack this
guard targets.
"""

import os

os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from db import get_connection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "data" / "processed" / "drift_reports"

FEATURE_COLS = ["district", "surface", "property_type"]

DATA_DEFINITION = DataDefinition(
    categorical_columns=["district", "property_type"],
    numerical_columns=["surface"],
)


def _query(conn, table: str, operation_type: str) -> pd.DataFrame:
    # Not pd.read_sql(conn=psycopg2 connection, ...) -- pandas warns that
    # DBAPI2 connections other than sqlite3 aren't officially supported;
    # a plain cursor + fetchall sidesteps that instead of just silencing it.
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(FEATURE_COLS)} FROM {table} WHERE operation_type = %s", (operation_type,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def load_reference(conn, operation_type: str) -> pd.DataFrame:
    return _query(conn, "features", operation_type)


def load_current(conn, operation_type: str) -> pd.DataFrame:
    return _query(conn, "prediction_logs", operation_type)


def build_report(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    report = Report(metrics=[DataDriftPreset()])
    return report.run(
        current_data=Dataset.from_pandas(current_df, data_definition=DATA_DEFINITION),
        reference_data=Dataset.from_pandas(reference_df, data_definition=DATA_DEFINITION),
    )


def save_report(snapshot, operation_type: str, reports_dir: Path = REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{operation_type.lower()}_drift.html"
    snapshot.save_html(str(path))
    return path


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    conn = get_connection()
    try:
        for operation_type in ["Venta", "Alquiler"]:
            reference_df = load_reference(conn, operation_type)
            current_df = load_current(conn, operation_type)
            if current_df.empty:
                print(f"{operation_type}: no rows in prediction_logs yet — skipping")
                continue
            snapshot = build_report(reference_df, current_df)
            path = save_report(snapshot, operation_type)
            print(
                f"{operation_type}: reference={len(reference_df)} rows, current={len(current_df)} rows. "
                f"Saved to {path.relative_to(REPO_ROOT)}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
