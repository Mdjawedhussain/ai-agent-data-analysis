"""Load any CSV into the SQLite database:  python load_data.py data/sample_sales.csv"""
import sys

from db import DB_PATH, get_schema, load_csv

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_sales.csv"
    n = load_csv(path)
    print(f"Loaded {n} rows into {DB_PATH}\n")
    print(get_schema())
