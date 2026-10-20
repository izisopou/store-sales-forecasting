# Store Sales - Time Series Forecasting

Predicting daily unit sales across 54 grocery stores and 33 product families for Kaggle's [Store Sales - Time Series Forecasting competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview).

This repository implements a production-grade demand forecasting pipeline using Google BigQuery and SQL for warehouse-level data extraction, Pandas for time-series feature engineering, and gradient boosted trees (XGBoost / LightGBM) for multi-step forecasting across a 16-day horizon.

> **Note on Model Scope:** This pipeline is specifically designed for short-term forecasting (16 days). For long-term forecasting (>6 months), a hybrid architecture is recommended (e.g., a linear or polynomial regressor to capture macro trends combined with a gradient boosting model trained on residuals).

---

## Environment Setup

```bash
# Clone repository
git clone git@github.com:izisopou/store-sales-forecasting.git
cd store-sales-forecasting

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Ingestion & Storage (BigQuery to Parquet)

The original competition data is provided by Kaggle as separate CSV files. To simulate a cloud data warehouse workflow, these tables were loaded into Google BigQuery under `store-sales-project.store_sales`.

While the competition tables are MB-scale and could technically fit in memory, heavy relational joins and filtering are delegated directly to BigQuery. This approach replicates a production-grade data warehouse pipeline and avoids repetitive, memory-heavy joins locally.

1. **BigQuery SQL Extraction:** 
   * Base tables (`train`, `test`, `stores`, `oil`, `holidays_events`) are merged and filtered directly in BigQuery using `sql/extract_base.sql`.
   * Workdays marked as holidays and duplicate event dates are filtered out in SQL to avoid row inflation.
   * `transactions` is extracted as a standalone table (`sql/extract_transactions.sql`) because store transactions are unavailable during the 16-day forecasting (test) window. Keeping it separate prevents accidental future data leakage and allows us to build safe lag features later.

2. **Local Parquet Storage:** 
   * Extracted data is saved locally in `data/interim/` as compressed `.parquet` files. Parquet preserves data types automatically and loads much faster than CSVs.

### Run Extraction

Ensure your Google Cloud credentials are authenticated, then execute:

```bash
# Extract all three tables (train, test, transactions) by default:
python src/extract_data.py

# Or specify individual tables to extract:
python src/extract_data.py train test
```

* **Source:** BigQuery dataset `store-sales-project.store_sales`
* **SQL Queries:** `sql/extract_base.sql` & `sql/extract_transactions.sql`
* **Outputs:** `train.parquet`, `test.parquet`, and `transactions.parquet` saved in `data/interim/`
* **Reference Notebook:** `notebooks/01_raw_data_inspection.ipynb`