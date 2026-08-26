"""
text_to_sql.py
--------------
Turns a plain-English question into a read-only SQL query against the
`sales` table, executes it safely, and returns a DataFrame.

Provider-agnostic: works with OpenAI, Anthropic, or any OpenAI-compatible
endpoint (Groq, Ollama, local LM Studio server, etc.) by setting the
LLM_PROVIDER / model env vars in .env. See .env.example.

Safety: only SELECT statements are allowed to run against the DB. Anything
else (INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA...) is rejected before
execution -- never trust generated SQL blindly.

Usage:
    from src.text_to_sql import ask
    df, sql = ask("What were total sales by region last quarter?")
"""

import os
import re
import sqlite3

import pandas as pd
from dotenv import load_dotenv

from src.data_prep import DB_PATH, get_schema_description

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a SQL generator for a SQLite database. Given a user question and \
the schema below, output ONLY a single valid, read-only SQLite SELECT statement that answers \
the question. Never modify data. Never use more than one statement. Do not wrap the SQL in \
markdown code fences -- output raw SQL only.

Schema:
{schema}
"""

_DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|CREATE|REPLACE|VACUUM)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(Exception):
    pass


def _validate_sql(sql: str) -> str:
    sql = sql.strip().strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    if not sql.lower().startswith("select"):
        raise UnsafeSQLError(f"Only SELECT statements are allowed. Got: {sql[:80]!r}")
    if _DISALLOWED.search(sql):
        raise UnsafeSQLError(f"Query contains a disallowed keyword: {sql[:120]!r}")
    if ";" in sql.strip().rstrip(";"):
        raise UnsafeSQLError("Only a single statement is allowed.")
    return sql


def generate_sql(question: str) -> str:
    """Calls the configured LLM provider to translate the question into SQL."""
    schema = get_schema_description()
    system_prompt = SYSTEM_PROMPT.format(schema=schema)

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        raw_sql = resp.choices[0].message.content
    elif LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        raw_sql = resp.content[0].text
    else:
        raise NotImplementedError(
            f"LLM_PROVIDER='{LLM_PROVIDER}' not wired up yet. "
            "Add a branch here for Ollama / local models -- same pattern."
        )

    return _validate_sql(raw_sql)


def run_sql(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def ask(question: str) -> tuple[pd.DataFrame, str]:
    """Full pipeline: question -> validated SQL -> executed -> DataFrame."""
    sql = generate_sql(question)
    df = run_sql(sql)
    return df, sql


# A few canned fallback queries so the Streamlit app has something to show
# even before an API key is configured (Phase 1, Week 3 checkpoint).
FALLBACK_QUERIES = {
    "total sales by region": "SELECT region, ROUND(SUM(sales), 2) AS total_sales FROM sales GROUP BY region ORDER BY total_sales DESC;",
    "total sales by category": "SELECT category, ROUND(SUM(sales), 2) AS total_sales FROM sales GROUP BY category ORDER BY total_sales DESC;",
    "monthly sales trend": "SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(sales), 2) AS total_sales FROM sales GROUP BY month ORDER BY month;",
    "top sub-categories by profit": "SELECT sub_category, ROUND(SUM(profit), 2) AS total_profit FROM sales GROUP BY sub_category ORDER BY total_profit DESC LIMIT 10;",
}

if __name__ == "__main__":
    for label, sql in FALLBACK_QUERIES.items():
        print(f"\n--- {label} ---")
        print(run_sql(sql))
