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

While the competition tables are MB-scale and can technically fit in memory, heavy relational joins and filtering are delegated directly to BigQuery. This approach replicates a production-grade data warehouse pipeline and avoids repetitive, memory-heavy joins locally.

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

# Dry-run to estimate scanned bytes without incurring GCP billing:
python src/extract_data.py --dry-run

# Extract specific targets with a custom billing cap:
python src/extract_data.py --targets train test --max-gb-billed 0.5
```

* **Source:** BigQuery dataset `store-sales-project.store_sales`
* **SQL Queries:** `sql/extract_base.sql` & `sql/extract_transactions.sql`
* **Outputs:** `train.parquet`, `test.parquet`, and `transactions.parquet` saved in `data/interim/`
* **Reference Notebook:** `notebooks/01_raw_data_inspection.ipynb`

## Feature Engineering Pipeline

The feature engineering stage processes interim tables into finalized feature matrices for model training and inference. To ensure reproducible, leak-free predictions across the 16-day test window, lags and rolling stats are computed on a continuous daily timeline across train and test sets to prevent data leakage and gaps.

1. **Calendar Completeness & Grid Validation:**
   * Validates that all 54 stores and 33 product families ($1,782$ series) are present on every recorded date.
   * Automatically detects closure dates (e.g., Christmas Day, Dec 25) missing from historical records and imputes them with zero sales so that day-based lags stay aligned with row shifts.
2. **Leak-Free Lag & Rolling Dynamics:**
   * Since predictions cover a 16-day horizon, target lags are strictly constrained to $\geq16$ days. Multiples of 7 (`lag21`, `lag28`, `lag35`) align with the exact day-of-week 3, 4, and 5 weeks prior.
   * Smoothed Weekday Baseline: `sales_dow_mean_3w` averages the last three aligned weekdays to capture a stable same-weekday baseline less sensitive to single-day spikes. 
   * Local Statistics: 14-day and 28-day rolling means and standard deviations over `sales_log1p_lag21` to capture short-term demand levels and volatility. 
   * Store Footfall: Store-level 16-day lagged transactions along with 14-day and 28-day rolling metrics to track store traffic trends without leakage.
3. **Promotions:**
   * Uses raw daily item promotion counts (`onpromotion`). Since promotion schedules are known in advance, they serve as a strong and direct forward-looking feature for the entire test window.
4. **Calendar & Payday Signals:**
   * Calendar indicators (`dayofweek`, `dayofmonth`, `dayofyear`, `month`, `year`) capture Ecuadorian retail patterns like weekend volume spikes ($+20\%$ to $+30\%$).
   * Binary flags for known store closures (`is_christmas`, `is_nyd`) and Ecuadorian paydays on the 15th and month-end (`is_mid_payday`, `is_month_start_end`).
5. **Warm-Up Trimming & Memory Optimization:**
   * The first two months (`2013-01-01` to `2013-02-28`) are used to warm up 35-day lags and rolling windows, and are dropped from training (`warmup_cutoff = 2013-03-01`).
   * High-cardinality columns (`store_nbr`, `family`, `city`, `state`, `store_type`, `cluster`, `holiday_name`) are cast to `category` on the combined dataset so category codes remain consistent across train and test.

### Excluded Features

* **`oil_price`:** Daily crude oil prices added noise without improving validation performance. Excluding raw oil prices yielded a lower (better) validation RMSLE. ***A potential next step is testing macro-level aggregations (e.g., 30-day rolling averages or month-over-month percentage changes) instead of noisy daily spot prices.***
* **`is_holiday` & `is_earthquake_period`:** Both indicators were dropped after ablation tests showed they were redundant, as their information is already fully captured and better split by the `holiday_name` categorical feature.

### Run Feature Engineering     

```bash
# Build features using default directories (data/interim -> data/processed):
python src/build_features.py

# Build features into a custom directory or specify a custom warm-up cutoff:
python src/build_features.py --output-dir data/processed --warmup-cutoff 2013-03-01
```

* **Inputs:** `train.parquet`, `test.parquet`, and `transactions.parquet` from `data/interim/`
* **Outputs:** `train_features.parquet` (35 columns) and `test_features.parquet` (33 columns) saved in `data/processed/`
* **Reference Notebook:** `notebooks/02_eda_fe.ipynb`