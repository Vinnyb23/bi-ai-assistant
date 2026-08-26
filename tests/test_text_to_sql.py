import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_to_sql import UnsafeSQLError, _validate_sql


def test_validate_sql_allows_select():
    sql = "SELECT region, SUM(sales) FROM sales GROUP BY region"
    assert _validate_sql(sql) == sql


def test_validate_sql_strips_markdown_fences():
    sql = "```sql\nSELECT * FROM sales\n```"
    result = _validate_sql(sql)
    assert result.lower().startswith("select")
    assert "```" not in result


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales WHERE 1=1",
        "UPDATE sales SET sales = 0",
        "INSERT INTO sales VALUES (1,2,3)",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA table_info(sales)",
    ],
)
def test_validate_sql_blocks_non_select(bad_sql):
    with pytest.raises(UnsafeSQLError):
        _validate_sql(bad_sql)


def test_validate_sql_blocks_multiple_statements():
    with pytest.raises(UnsafeSQLError):
        _validate_sql("SELECT * FROM sales; DROP TABLE sales;")
