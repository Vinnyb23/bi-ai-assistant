"""
data_prep.py
------------
Generates a realistic synthetic retail-sales dataset and loads it into a
local SQLite database (data/bi_assistant.db). This stands in for a real
BI data warehouse export so the whole project runs with zero external
dependencies or API keys.

Swap this out later for a real extract (Kaggle "Superstore", your own
SQL Server / Snowflake export, etc.) -- the rest of the project only
depends on the `sales` table schema defined below.

Usage:
    python src/data_prep.py
"""

import os
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bi_assistant.db")
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
CATEGORIES = ["Furniture", "Office Supplies", "Technology"]
SUB_CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Bookcases"],
    "Office Supplies": ["Paper", "Binders", "Storage"],
    "Technology": ["Phones", "Accessories", "Machines"],
}


def generate_sales_data(start="2022-01-01", end="2026-08-01", seed=42) -> pd.DataFrame:
    """Create a daily-grain synthetic sales table with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")

    rows = []
    order_id = 100000
    for date in dates:
        # More orders on weekdays, seasonal bump in Nov/Dec, mild YoY growth.
        day_factor = 1.3 if date.weekday() < 5 else 0.8
        season_factor = 1.6 if date.month in (11, 12) else 1.0
        growth_factor = 1 + 0.15 * ((date.year - 2022) + date.month / 12)
        n_orders = int(rng.poisson(lam=6 * day_factor * season_factor * growth_factor))

        for _ in range(n_orders):
            region = rng.choice(REGIONS)
            category = rng.choice(CATEGORIES)
            sub_category = rng.choice(SUB_CATEGORIES[category])
            base_price = {"Furniture": 220, "Office Supplies": 35, "Technology": 310}[category]
            quantity = int(rng.integers(1, 6))
            unit_price = max(5, rng.normal(base_price, base_price * 0.25))
            discount = float(rng.choice([0, 0, 0.1, 0.15, 0.2], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
            sales = round(unit_price * quantity * (1 - discount), 2)
            profit = round(sales * rng.normal(0.18, 0.08), 2)

            order_id += 1
            rows.append(
                {
                    "order_id": order_id,
                    "order_date": date.strftime("%Y-%m-%d"),
                    "region": region,
                    "category": category,
                    "sub_category": sub_category,
                    "quantity": quantity,
                    "unit_price": round(unit_price, 2),
                    "discount": discount,
                    "sales": sales,
                    "profit": profit,
                }
            )

    return pd.DataFrame(rows)


def load_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("sales", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(order_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region);")
        conn.commit()
    finally:
        conn.close()


def get_schema_description(db_path: str = DB_PATH) -> str:
    """Human-readable schema description fed to the LLM for text-to-SQL grounding."""
    return (
        "Table: sales\n"
        "Columns:\n"
        "  order_id INTEGER -- unique order identifier\n"
        "  order_date TEXT  -- ISO date 'YYYY-MM-DD'\n"
        "  region TEXT      -- one of: Northeast, Southeast, Midwest, West, Southwest\n"
        "  category TEXT    -- one of: Furniture, Office Supplies, Technology\n"
        "  sub_category TEXT\n"
        "  quantity INTEGER\n"
        "  unit_price REAL\n"
        "  discount REAL    -- fraction, e.g. 0.15 = 15%\n"
        "  sales REAL       -- net revenue for the line item\n"
        "  profit REAL\n"
    )


if __name__ == "__main__":
    print("Generating synthetic sales data...")
    df = generate_sales_data()
    print(f"Generated {len(df):,} order line items from {df.order_date.min()} to {df.order_date.max()}")
    load_to_sqlite(df)
    print(f"Loaded into {DB_PATH}")
