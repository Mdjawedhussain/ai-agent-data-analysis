
"""SQLite helpers: load a CSV, describe the schema, and run read-only, validated SQL."""

import os
import re
import sqlite3

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/analytics.db")
TABLE = "sales"
CSV_PATH = os.getenv("CSV_PATH", "data/sales.csv")

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|truncate|vacuum|reindex)\b",
    re.I,
)


def load_csv(csv_path: str, table: str = TABLE, db_path: str = DB_PATH) -> int:
    """Load a CSV into SQLite, creating or replacing the table."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            "Set CSV_PATH in your .env file to the correct path."
        )

    df = pd.read_csv(csv_path)

    if df.empty and len(df.columns) == 0:
        raise ValueError(f"CSV contains no columns: {csv_path}")

    df.columns = [
        re.sub(r"\W+", "_", c.strip()).strip("_").lower()
        for c in df.columns
    ]

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    with sqlite3.connect(db_path) as con:
        df.to_sql(table, con, if_exists="replace", index=False)

    return len(df)


def _table_exists(con, table: str) -> bool:
    """Check whether a table exists in SQLite."""
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def ensure_table(db_path: str = DB_PATH, table: str = TABLE) -> None:
    """Create the database directory and initialize the table from CSV if needed."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    with sqlite3.connect(db_path) as con:
        if _table_exists(con, table):
            return

    # Table does not exist; initialize it from the CSV.
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"Table '{table}' does not exist in {db_path}, and CSV file "
            f"was not found at {CSV_PATH}. Put your CSV there or set "
            "CSV_PATH in your .env file."
        )

    load_csv(CSV_PATH, table, db_path)


def get_schema(db_path: str = DB_PATH, table: str = TABLE) -> str:
    """Return table schema and sample rows, initializing the table if necessary."""
    ensure_table(db_path, table)

    with sqlite3.connect(db_path) as con:
        # Parameterize the table lookup and safely quote its identifier.
        safe_table = '"' + table.replace('"', '""') + '"'

        cols = con.execute(
            f"PRAGMA table_info({safe_table})"
        ).fetchall()

        sample = pd.read_sql_query(
            f"SELECT * FROM {safe_table} LIMIT 3", con
        )

        n = con.execute(
            f"SELECT COUNT(*) FROM {safe_table}"
        ).fetchone()[0]

    col_lines = "\n".join(
        f"  - {c[1]} ({c[2] or 'TEXT'})" for c in cols
    )

    return (
        f"Table: {table} ({n} rows)\n"
        f"Columns:\n{col_lines}\n"
        f"Sample rows:\n{sample.to_csv(index=False)}"
    )


def validate_sql(sql: str) -> str:
    """Allow a single read-only SELECT. Returns cleaned SQL or raises ValueError."""
    s = (sql or "").strip().rstrip(";").strip()

    if not s:
        raise ValueError("Empty SQL")

    if ";" in s:
        raise ValueError("Only one statement is allowed")

    if not re.match(r"^(select|with)\b", s, re.I):
        raise ValueError("Only SELECT queries are allowed")

    if FORBIDDEN.search(s):
        raise ValueError("Query contains a forbidden keyword")

    return s


def run_query(
    sql: str, db_path: str = DB_PATH, max_rows: int = 500
) -> pd.DataFrame:
    """Run a validated read-only SQL query."""
    s = validate_sql(sql)

    con = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)

    try:
        return pd.read_sql_query(s, con).head(max_rows)
    finally:
        con.close()
