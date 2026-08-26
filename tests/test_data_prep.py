import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import generate_sales_data, load_to_sqlite


def test_generate_sales_data_has_expected_columns():
    df = generate_sales_data(start="2024-01-01", end="2024-01-31")
    expected_cols = {
        "order_id", "order_date", "region", "category", "sub_category",
        "quantity", "unit_price", "discount", "sales", "profit",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) > 0


def test_generate_sales_data_values_are_sane():
    df = generate_sales_data(start="2024-01-01", end="2024-03-31")
    assert (df["quantity"] >= 1).all()
    assert (df["unit_price"] > 0).all()
    assert df["region"].nunique() <= 5
    assert df["category"].nunique() <= 3


def test_load_to_sqlite_creates_queryable_table(tmp_path):
    df = generate_sales_data(start="2024-01-01", end="2024-01-15")
    db_path = tmp_path / "test.db"
    load_to_sqlite(df, db_path=str(db_path))

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    finally:
        conn.close()
    assert count == len(df)
