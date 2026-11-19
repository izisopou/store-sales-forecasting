import argparse
from pathlib import Path
import db_dtypes
import numpy as np
import pandas as pd
from pyprojroot import here

PROJ_PATH = here()

DEFAULT_INPUT_PATH = PROJ_PATH / 'data' / 'interim'
DEFAULT_OUTPUT_PATH = PROJ_PATH / 'data' / 'processed'
DEFAULT_WARMUP_CUTOFF = '2013-03-01'

CATEGORICAL_COLS = [
    'store_nbr',
    'family',
    'city',
    'state',
    'store_type',
    'cluster',
    'holiday_name',
]

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    examples = """Examples:
      # Build features using default directories (data/interim -> data/processed):
      python src/build_features.py

      # Build features into a temporary directory for cross-checks and/or change the warmup cutoff:
      python src/build_features.py --output-dir tmp --warmup-cutoff 2013-04-01
    """  
    parser = argparse.ArgumentParser(
        description='Clean interim datasets and build leak-free features for modeling.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f'Input directory containing interim parquet files (default: {DEFAULT_INPUT_PATH}).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Output directory to save processed feature parquets (default: {DEFAULT_OUTPUT_PATH}).',
    )
    parser.add_argument(
        '--warmup-cutoff',
        type=str,
        default=DEFAULT_WARMUP_CUTOFF,
        help=f'Earliest date for training samples after feature warm-up saturation (default: {DEFAULT_WARMUP_CUTOFF}).',
    )
    return parser.parse_args()


def clean_datasets(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    df_transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean input datasets, drop unused raw columns, validate grid completeness,
    and insert missing full-closure dates (e.g. Christmas Day) in train.
    """    
    # 1. Drop raw columns excluded in the modeling (is_holiday)
    cols_to_drop = ['is_holiday']
    df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])

    # 2. Convert date columns to datetime and sort chronologically
    df_train = (
        df_train
        .assign(date=lambda x: pd.to_datetime(x['date']))
        .sort_values(by=['store_nbr', 'family', 'date'])
        .reset_index(drop=True)
    )
    df_test = (
        df_test
        .assign(date=lambda x: pd.to_datetime(x['date']))
        .sort_values(by=['store_nbr', 'family', 'date'])
        .reset_index(drop=True)
    )
    df_transactions = (
        df_transactions
        .assign(date=lambda x: pd.to_datetime(x['date']))
        .sort_values(by=['store_nbr', 'date'])
        .reset_index(drop=True)
    )  

    # 3. Validate store-family completeness per recorded date in train
    n_stores = df_train['store_nbr'].nunique()
    n_families = df_train['family'].nunique()
    expected_series_per_day = n_stores * n_families

    train_rows_per_date = df_train.groupby('date').size()
    incomplete_train_dates = train_rows_per_date[train_rows_per_date != expected_series_per_day]
    if not incomplete_train_dates.empty:
        raise ValueError(
            f'Corrupted train data: {len(incomplete_train_dates)} dates have incomplete '
            f'(store_nbr, family) pairs: {list(incomplete_train_dates.index.strftime("%Y-%m-%d"))}'
        )

    # 4. Detect and insert missing full calendar dates in train (e.g., Christmas Day closures)
    full_train_calendar = pd.date_range(
        start=df_train['date'].min(),
        end=df_train['date'].max(),
        freq='D',
    )
    missing_dates = full_train_calendar.difference(df_train['date'].unique())

    if len(missing_dates) > 0:
        print(
            f'Inserting {len(missing_dates)} missing closure date(s) in train: '
            f'{[str(d.date()) for d in missing_dates]}'
        )
        store_family_meta = (
            df_train[['store_nbr', 'family', 'city', 'state', 'store_type', 'cluster']]
            .drop_duplicates()
            .sort_values(by=['store_nbr', 'family'])
            .reset_index(drop=True)
        )
        missing_dates_df = (
            pd.DataFrame({'date': missing_dates})
            .merge(store_family_meta, how='cross')
            .assign(
                id=-1,
                onpromotion=0,
                oil_price=np.nan,
                sales=0.0,
                holiday_name=lambda x: np.where(
                    (x['date'].dt.month == 12) & (x['date'].dt.day == 25),
                    'Navidad',
                    'Unknown Closure',
                ),
            )[df_train.columns]
            .astype(df_train.dtypes.to_dict())
        )
        df_train = (
            pd.concat([df_train, missing_dates_df], ignore_index=True)
            .sort_values(by=['store_nbr', 'family', 'date'])
            .reset_index(drop=True)
        )

    # 5. Validate test set calendar and grid integrity
    full_test_calendar = pd.date_range(
        start=df_test['date'].min(),
        end=df_test['date'].max(),
        freq='D',
    )
    if len(df_test) != len(full_test_calendar) * expected_series_per_day:
        raise ValueError(
            'Test dataset does not form a complete (date x store_nbr x family) grid.'
        )

    # 6. Fill non-holiday NaNs in holiday_name with 'None'
    df_train['holiday_name'] = df_train['holiday_name'].fillna('None')
    df_test['holiday_name'] = df_test['holiday_name'].fillna('None')  

    # 7. Impute missing oil_price values across the full timeline and downcast to float32
    oil_dates = (
        pd.concat([df_train[['date', 'oil_price']], df_test[['date', 'oil_price']]])
        .drop_duplicates(subset='date')
        .sort_values('date')
        .ffill()
        .bfill()
    )
    oil_map = oil_dates.set_index('date')['oil_price']
    df_train['oil_price'] = df_train['date'].map(oil_map).astype(np.float32)
    df_test['oil_price'] = df_test['date'].map(oil_map).astype(np.float32)

    df_train['onpromotion'] = df_train['onpromotion'].astype(np.float32)
    df_test['onpromotion'] = df_test['onpromotion'].astype(np.float32)  

    return df_train, df_test, df_transactions
   

def add_calendar_and_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar, payday, and major closure flags."""
    return df.assign(
        year = lambda x: x['date'].dt.year.astype('int16'),
        month = lambda x: x['date'].dt.month.astype('int8'),
        dayofmonth = lambda x: x['date'].dt.day.astype('int8'),
        dayofweek = lambda x: x['date'].dt.dayofweek.astype('int8'),
        dayofyear = lambda x: x['date'].dt.dayofyear.astype('int16'),
        is_nyd = lambda x: (x['date'].dt.dayofyear == 1).astype('int8'),
        is_christmas = lambda x: ((x['date'].dt.month == 12) & (x['date'].dt.day == 25)).astype('int8'),
        is_month_start_end = lambda x: ((x['date'].dt.day <= 4) | (x['date'].dt.is_month_end)).astype('int8'),
        is_mid_payday = lambda x: x['date'].dt.day.isin([15, 16]).astype('int8'),
    )

def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create promotional lags, rolling intensity, and signed/relative momentum features."""
    return (
        df
        .sort_values(by=['store_nbr', 'family', 'date'])
        .assign(
            onpromotion_lag7 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].shift(7).astype(np.float32),
            onpromotion_lag16 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].shift(16).astype(np.float32),
            onpromotion_lag21 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].shift(21).astype(np.float32),
            onpromotion_rolling_mean_7 = lambda x: (
                x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].transform(lambda s: s.rolling(window=7, min_periods=1).mean())
            ).astype(np.float32),
            onpromotion_rolling_std_7 = lambda x: (
                x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].transform(lambda s: s.rolling(window=7, min_periods=1).std())
            ).fillna(0.0).astype(np.float32),
            onpromotion_rolling_mean_14 = lambda x: (
                x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].transform(lambda s: s.rolling(window=14, min_periods=1).mean())
            ).astype(np.float32),
            onpromotion_rolling_std_14 = lambda x: (
                x.groupby(['store_nbr', 'family'], sort=False)['onpromotion'].transform(lambda s: s.rolling(window=14, min_periods=1).std())
            ).fillna(0.0).astype(np.float32),
            promo_diff_lag7 = lambda x: (x['onpromotion'] - x['onpromotion_lag7']).astype(np.float32),
            promo_diff_lag16 = lambda x: (x['onpromotion'] - x['onpromotion_lag16']).astype(np.float32),
            promo_diff_lag21 = lambda x: (x['onpromotion'] - x['onpromotion_lag21']).astype(np.float32),
            promo_diff_mean_7 = lambda x: (x['onpromotion'] - x['onpromotion_rolling_mean_7']).astype(np.float32),
            promo_rel_diff_mean_7 = lambda x: (
                (x['onpromotion'] - x['onpromotion_rolling_mean_7']) / (x['onpromotion_rolling_mean_7'] + 1.0)
            ).astype(np.float32),
            promo_rel_diff_mean_14 = lambda x: (
                (x['onpromotion'] - x['onpromotion_rolling_mean_14']) / (x['onpromotion_rolling_mean_14'] + 1.0)
            ).astype(np.float32),
        )
    )

def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create macroeconomic oil features: 28-day rolling stats and MoM momentum."""
    daily_oil = (
        df[['date', 'oil_price']]
        .drop_duplicates(subset='date')
        .sort_values('date')
        .reset_index(drop=True)
        .assign(
            oil_price_rolling_mean_28=lambda x: x['oil_price'].rolling(window=28, min_periods=1).mean().astype(np.float32),
            oil_price_rolling_std_28=lambda x: x['oil_price'].rolling(window=28, min_periods=1).std().fillna(0.0).astype(np.float32),
            oil_price_lag28=lambda x: x['oil_price'].shift(28),
            oil_price_lag28_rolling_mean_28=lambda x: x['oil_price_rolling_mean_28'].shift(28),
            oil_price_dev_ma28_pct=lambda x: (
                (x['oil_price'] - x['oil_price_rolling_mean_28'])
                / x['oil_price_rolling_mean_28'].replace(0, np.nan)
            ).fillna(0.0).astype(np.float32),
            oil_price_point_mom_pct=lambda x: (
                (x['oil_price'] - x['oil_price_lag28'])
                / x['oil_price_lag28'].replace(0, np.nan)
            ).fillna(0.0).astype(np.float32),
            oil_price_trend_mom_pct=lambda x: (
                (x['oil_price_rolling_mean_28'] - x['oil_price_lag28_rolling_mean_28'])
                / x['oil_price_lag28_rolling_mean_28'].replace(0, np.nan)
            ).fillna(0.0).astype(np.float32),
        )
        .drop(columns=['oil_price_lag28', 'oil_price_lag28_rolling_mean_28', 'oil_price'])
    )
    return df.merge(daily_oil, on='date', how='left')

def add_target_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Create the log1p-transformed target variable for RMSLE alignment."""
    return df.assign(sales_log1p = lambda x: np.log1p(x['sales']).astype(np.float32))

def add_sales_log1p_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free short-term lags, rolling/EWMA metrics, annual lags, and hierarchical aggregations."""
    return (
        df
        .sort_values(by=['store_nbr', 'family', 'date'])
        .assign(
            # 1. Point-in-time lags & 3-week aligned weekday baseline
            sales_log1p_lag16 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p'].shift(16).astype(np.float32),
            sales_log1p_lag21 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p'].shift(21).astype(np.float32),
            sales_log1p_lag28 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p'].shift(28).astype(np.float32),
            sales_log1p_lag35 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p'].shift(35).astype(np.float32),
            sales_dow_mean_3w = lambda df: ((df['sales_log1p_lag21'] + df['sales_log1p_lag28'] + df['sales_log1p_lag35']) / 3.0).astype(np.float32),
            # 2. Rolling statistics over lag21 (v1.0.0)
            sales_log1p_lag21_rolling_mean_14 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag21'].transform(
                lambda s: s.rolling(window=14, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_lag21_rolling_mean_28 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag21'].transform(
                lambda s: s.rolling(window=28, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_lag21_rolling_std_14 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag21'].transform(
                lambda s: s.rolling(window=14, min_periods=1).std()
            ).fillna(0.0).astype(np.float32),
            sales_log1p_lag21_rolling_std_28 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag21'].transform(
                lambda s: s.rolling(window=28, min_periods=1).std()
            ).fillna(0.0).astype(np.float32),
            # 3. Rolling & EWMA statistics over lag16 + log-space momentum (v1.1.0)
            sales_log1p_lag16_rolling_mean_7 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag16'].transform(
                lambda s: s.rolling(window=7, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_lag16_rolling_mean_14 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag16'].transform(
                lambda s: s.rolling(window=14, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_lag16_rolling_std_7 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag16'].transform(
                lambda s: s.rolling(window=7, min_periods=1).std()
            ).fillna(0.0).astype(np.float32),
            sales_log1p_lag16_rolling_std_14 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag16'].transform(
                lambda s: s.rolling(window=14, min_periods=1).std()
            ).fillna(0.0).astype(np.float32),
            sales_log1p_lag16_ewm_mean_7 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag16'].transform(
                lambda s: s.ewm(span=7, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_lag16_ewm_mean_14 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag16'].transform(
                lambda s: s.ewm(span=14, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_momentum_7_28 = lambda x: (x['sales_log1p_lag16_rolling_mean_7'] - x['sales_log1p_lag21_rolling_mean_28']).astype(np.float32),
            # 4. Weekday-aligned 52-week (364-day) annual seasonal lags (v1.1.0)
            sales_log1p_lag364 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p'].shift(364).astype(np.float32),
            sales_log1p_lag364_rolling_mean_7 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag364'].transform(
                lambda s: s.rolling(window=7, min_periods=1).mean()
            ).astype(np.float32),
            sales_log1p_lag364_rolling_std_7 = lambda x: x.groupby(['store_nbr', 'family'], sort=False)['sales_log1p_lag364'].transform(
                lambda s: s.rolling(window=7, min_periods=1).std()
            ).fillna(0.0).astype(np.float32),
            # 5. Hierarchical Target Aggregations: Store-level & Regional (State x Family) (v1.1.0)
            store_sales_log1p_lag16_rolling_mean_7 = lambda x: x.groupby(['date', 'store_nbr'], sort=False)['sales_log1p_lag16_rolling_mean_7'].transform('mean').astype(np.float32),
            state_family_sales_log1p_lag16_rolling_mean_7 = lambda x: x.groupby(['date', 'state', 'family'], sort=False)['sales_log1p_lag16_rolling_mean_7'].transform('mean').astype(np.float32),
        )
    )

def add_transactions_lag_features(df: pd.DataFrame, df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free 16/21/28/35-day lag and rolling/EWM transaction features."""
    all_dates = pd.date_range(
        start=df['date'].min(),
        end=df['date'].max(),
        freq='D',
    )   
    all_stores = df_transactions['store_nbr'].unique()

    full_grid = (
        pd.MultiIndex.from_product(
            [all_dates, all_stores], names=['date', 'store_nbr']
        )
        .to_frame()
        .reset_index(drop=True)
    )

    df_transactions_complete = (
        pd.merge(
            full_grid,
            df_transactions,
            on=['date', 'store_nbr'],
            how='left',
        )
        .sort_values(by=['store_nbr', 'date'])
        .fillna(0)
        .assign(
            transactions_lag16 = lambda df: df.groupby('store_nbr')['transactions'].shift(16).astype(np.float32),
            transactions_lag21 = lambda df: df.groupby('store_nbr')['transactions'].shift(21).astype(np.float32),
            transactions_lag28 = lambda df: df.groupby('store_nbr')['transactions'].shift(28).astype(np.float32),
            transactions_lag35 = lambda df: df.groupby('store_nbr')['transactions'].shift(35).astype(np.float32),
            transactions_lag16_rolling_mean_7 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.rolling(7, min_periods=1).mean())
                .astype(np.float32)
            ),
            transactions_lag16_rolling_mean_14 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.rolling(14, min_periods=1).mean())
                .astype(np.float32)
            ),
            transactions_lag16_rolling_mean_28 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.rolling(28, min_periods=1).mean())
                .astype(np.float32)
            ),
            transactions_lag16_rolling_std_7 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.rolling(7, min_periods=1).std())
                .fillna(0.0)
                .astype(np.float32)
            ),
            transactions_lag16_rolling_std_14 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.rolling(14, min_periods=1).std())
                .fillna(0.0)
                .astype(np.float32)
            ),
            transactions_lag16_rolling_std_28 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.rolling(28, min_periods=1).std())
                .fillna(0.0)
                .astype(np.float32)
            ),
            transactions_lag16_ewm_mean_7 = lambda df: (
                df.groupby('store_nbr')['transactions_lag16']
                .transform(lambda s: s.ewm(span=7, min_periods=1).mean())
                .astype(np.float32)
            ),
        )
    )

    return df.merge(
        df_transactions_complete,
        on=['date', 'store_nbr'],
        how='left',
    ).drop(columns=['transactions'])

def main() -> None:
    """Main execution entry point for cleaning data and building features."""
    args = parse_args()

    print('Loading interim parquet datasets...')
    df_train = pd.read_parquet(args.input_dir / 'train.parquet')
    df_test = pd.read_parquet(args.input_dir / 'test.parquet')
    df_transactions = pd.read_parquet(args.input_dir / 'transactions.parquet')

    print('Cleaning datasets and verifying timeline completeness...')
    clean_df_train, clean_df_test, clean_df_transactions = clean_datasets(
        df_train=df_train,
        df_test=df_test,
        df_transactions=df_transactions,
    )

    train_max_date = clean_df_train['date'].max()

    print('Engineering time-series and calendar features...')
    df_full = (
        pd.concat([clean_df_train, clean_df_test], ignore_index=True, sort=False)
        .sort_values(by=['store_nbr', 'family', 'date'])
        .pipe(add_transactions_lag_features, df_transactions=clean_df_transactions)
        .pipe(add_calendar_and_event_features)
        .pipe(add_promotion_features)
        .pipe(add_oil_features)
        .pipe(add_target_log_transform)
        .pipe(add_sales_log1p_lag_features)
    )

    for col in CATEGORICAL_COLS:
        df_full[col] = df_full[col].astype('str').astype('category')

    df_train_processed = (
        df_full
        .loc[(df_full['date'] >= args.warmup_cutoff) & (df_full['date'] <= train_max_date)]
        .reset_index(drop=True)
    )

    df_test_processed = (
        df_full
        .loc[df_full['date'] > train_max_date]
        .drop(columns=['sales', 'sales_log1p'])
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file_train = args.output_dir / 'train_features.parquet'
    output_file_test = args.output_dir / 'test_features.parquet'

    df_train_processed.to_parquet(output_file_train, index=False)
    print(
        f'Saved {output_file_train.name} '
        f'({df_train_processed.shape[0]:,} rows, {df_train_processed.shape[1]} cols) successfully.'
    )

    df_test_processed.to_parquet(output_file_test, index=False)
    print(
        f'Saved {output_file_test.name} '
        f'({df_test_processed.shape[0]:,} rows, {df_test_processed.shape[1]} cols) successfully.'
    )


if __name__ == '__main__':
    main()