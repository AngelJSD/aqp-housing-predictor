"""Shared Postgres connection helper for the loader scripts in this
directory — factored out once a second script needed the identical
host/port/credentials logic, to avoid the two copies drifting apart.
"""

import os

import psycopg2


def get_connection():
    """Connects from the host (not inside the Compose network), so it
    targets localhost + the mapped host port, not the `postgres` service
    name."""
    return psycopg2.connect(
        host="localhost",
        port=os.environ["POSTGRES_HOST_PORT"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
