# BI Copilot — Natural-Language BI Assistant

**Phase 1 flagship project** of a 6-month self-directed AI/ML continuing-education program (following the UT Austin PGP-AI certificate). This phase combines 15 years of BI/SQL experience with ML to ship a "BI + AI fusion" tool: ask a plain-English question, get a live SQL query and chart back, plus a 30-day sales forecast — no manual report-building required.

> Live demo: _add your Streamlit Community Cloud / Hugging Face Spaces link here after deploying (Week 6)_

![Python](https://img.shields.io/badge/python-3.11-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![MLflow](https://img.shields.io/badge/tracked%20with-MLflow-0194E2)

## What it does

- **Forecasting** — an XGBoost model predicts the next 30 days of total sales from calendar + lag/rolling features, with every run tracked in MLflow (params, metrics, model artifact).
- **Text-to-SQL** — a plain-English question ("What were total sales by region last quarter?") is turned into a validated, read-only SQL query by an LLM, executed against a real SQLite database, and rendered as a table + chart.
- **Safety guardrail** — generated SQL is checked before execution: only single `SELECT` statements are allowed; anything else (`DROP`, `DELETE`, `PRAGMA`, multiple statements, etc.) is rejected. See `src/text_to_sql.py::_validate_sql`.
- **BI-style dashboard** — a Streamlit front end ties the forecast and the Q&A box together in one app.

## Architecture

```
┌─────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│  data_prep.py    │ ---> │  SQLite (sales)    │ <--- │  text_to_sql.py    │
│  synthetic data  │      │  bi_assistant.db   │      │  LLM -> validated  │
└─────────────────┘      └───────────────────┘      │  SQL -> DataFrame  │
                                    │                 └─────────┬─────────┘
                                    v                            │
                           ┌───────────────────┐                │
                           │  forecast.py       │                │
                           │  XGBoost + MLflow  │                │
                           └─────────┬─────────┘                │
                                     │                            │
                                     v                            v
                              ┌────────────────────────────────────┐
                              │              app.py                 │
                              │         Streamlit dashboard         │
                              └────────────────────────────────────┘
```

## Results (backtest on synthetic data)

| Metric | Value |
|---|---|
| MAE (last ~15% holdout) | ~$2,100 / day |
| MAPE | ~44% |
| Forecast horizon | 30 days |

_This is a deliberately simple baseline (XGBoost + lag features on synthetic data with injected noise) — the point of Week 2 is to beat this number and log the comparison in MLflow, not to ship a production-tuned model on day one._

## Project structure

```
bi-ai-assistant/
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── data/                 <- generated at runtime (gitignored): SQLite DB, saved model
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── data_prep.py      <- synthetic data generator + SQLite loader
│   ├── forecast.py       <- XGBoost forecasting + MLflow logging
│   ├── text_to_sql.py    <- LLM-backed text-to-SQL with safety validation
│   └── app.py            <- Streamlit app
└── tests/
    ├── test_data_prep.py
    ├── test_forecast.py
    └── test_text_to_sql.py
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/bi-ai-assistant.git
cd bi-ai-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) configure an LLM for text-to-SQL

```bash
cp .env.example .env
# then edit .env and add OPENAI_API_KEY or ANTHROPIC_API_KEY
```

Without an API key, the app still runs fully — the "Ask Your Data" tab falls back to a set of pre-built sample questions that execute real SQL against the database.

### 3. Generate the sample data

```bash
python src/data_prep.py
```

### 4. Train the forecast model (optional standalone run)

```bash
python src/forecast.py
```

### 5. Launch the dashboard

```bash
streamlit run src/app.py
```

### 6. Inspect experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### Run with Docker instead

```bash
docker build -t bi-ai-assistant .
docker run -p 8501:8501 --env-file .env bi-ai-assistant
```

### Run the tests

```bash
pytest tests/ -v
```

## Swapping in real data

Everything downstream only depends on the `sales` table schema (see `src/data_prep.py::get_schema_description`). To use your own BI export instead of the synthetic generator: load your data into the same schema (or update the schema description + queries), point `DB_PATH` at your database, and the forecasting + text-to-SQL layers work unchanged.

## Roadmap for this repo (Phase 1, Weeks 1–6)

- [x] Week 1: synthetic data generator + SQLite schema
- [x] Week 2: XGBoost forecasting baseline logged to MLflow
- [x] Week 3: text-to-SQL layer with safety validation
- [x] Week 4: Streamlit dashboard combining both
- [ ] Week 5: expand test coverage, containerize (Dockerfile above is a first pass)
- [ ] Week 6: deploy a live demo (Hugging Face Spaces or Streamlit Community Cloud), add screenshots below, write the retrospective

## Screenshots

_Add a screenshot or GIF of the Forecast tab and the Ask Your Data tab here once you run the app (Week 6 polish task)._

## Part of a larger program

This repo is Phase 1 of a 6-month self-directed AI/ML program:

1. **BI + AI Fusion** — this repo
2. Computer Vision — `explainable-vision-classifier`
3. Generative AI & Agents — `ai-bi-analyst-agent`
4. MLOps & Deployment (capstone) — `bi-copilot-capstone`, which unifies all three
