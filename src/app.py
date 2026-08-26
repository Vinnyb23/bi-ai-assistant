"""
app.py
------
Streamlit dashboard combining:
  1. A 30-day sales forecast chart (src/forecast.py)
  2. A natural-language "ask your data" box that turns questions into SQL
     and shows the resulting table + chart (src/text_to_sql.py)

Run:
    streamlit run src/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from src.data_prep import DB_PATH, generate_sales_data, load_to_sqlite
from src.forecast import run_pipeline
from src.text_to_sql import FALLBACK_QUERIES, UnsafeSQLError, ask, run_sql

st.set_page_config(page_title="BI Copilot", page_icon="📊", layout="wide")

st.title("📊 BI Copilot -- Natural-Language BI Assistant")
st.caption(
    "Phase 1 flagship project: ask questions in plain English and see a live "
    "30-day sales forecast, all backed by a real SQL database."
)

if not os.path.exists(DB_PATH):
    with st.spinner("First run: generating sample data..."):
        load_to_sqlite(generate_sales_data())

tab_forecast, tab_ask, tab_about = st.tabs(["📈 Forecast", "💬 Ask Your Data", "ℹ️ About"])

with tab_forecast:
    st.subheader("30-Day Sales Forecast")
    with st.spinner("Training forecast model (XGBoost, logged to MLflow)..."):
        daily, future, metrics = run_pipeline()

    col1, col2, col3 = st.columns(3)
    col1.metric("Backtest MAE", f"${metrics['mae']:,.0f}")
    col2.metric("Backtest MAPE", f"{metrics['mape']:.1%}")
    col3.metric("Forecast horizon", "30 days")

    history_tail = daily.tail(90).rename(columns={"total_sales": "Actual"}).set_index("order_date")
    forecast_df = future.rename(columns={"total_sales": "Forecast"}).set_index("order_date")
    chart_df = pd.concat([history_tail[["Actual"]], forecast_df[["Forecast"]]], axis=1)
    st.line_chart(chart_df)
    st.caption(
        "Last 90 days of actuals vs. the next 30 days forecast. "
        "Model runs are tracked in MLflow (`mlruns.db`) -- run `mlflow ui "
        "--backend-store-uri sqlite:///mlruns.db` to inspect experiments."
    )

with tab_ask:
    st.subheader("Ask a question about your sales data")
    st.caption(
        "Type a question in plain English. If no LLM API key is configured yet, "
        "pick one of the sample questions below -- they run against the real database."
    )

    sample_choice = st.selectbox("Sample questions", ["(type my own)"] + list(FALLBACK_QUERIES.keys()))
    question = st.text_input("Or type your own question", placeholder="e.g. What were total sales by region last quarter?")

    if st.button("Run query", type="primary"):
        try:
            if sample_choice != "(type my own)" and not question:
                sql = FALLBACK_QUERIES[sample_choice]
                result_df = run_sql(sql)
            elif question:
                result_df, sql = ask(question)
            else:
                st.warning("Pick a sample question or type your own.")
                st.stop()

            st.code(sql, language="sql")
            st.dataframe(result_df, use_container_width=True)

            numeric_cols = result_df.select_dtypes("number").columns
            if len(numeric_cols) and len(result_df) > 1:
                st.bar_chart(result_df.set_index(result_df.columns[0])[numeric_cols])

        except UnsafeSQLError as e:
            st.error(f"Blocked an unsafe query: {e}")
        except NotImplementedError as e:
            st.warning(f"{e}\n\nUsing a sample question instead is fine while you set up an LLM_PROVIDER.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Something went wrong: {e}")

with tab_about:
    st.markdown(
        """
### About this project
This is the **Phase 1** flagship project of a 6-month self-directed AI/ML
continuing-education program (BI + AI Fusion phase). It combines:

- **Forecasting**: XGBoost with calendar + lag features, experiment-tracked in MLflow
- **Text-to-SQL**: an LLM translates plain-English questions into safe, read-only SQL
- **BI-style dashboarding**: Streamlit front end

See the project [README](README.md) for setup, architecture, and roadmap details.
        """
    )
