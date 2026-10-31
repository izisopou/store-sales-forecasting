import argparse
from pathlib import Path
import db_dtypes
import joblib
import json
import numpy as np
import pandas as pd
from pyprojroot import here
import xgboost as xgb
import lightgbm as lgb

PROJ_PATH = here()

DEFAULT_MODEL_PATH = PROJ_PATH / 'models'
DEFAULT_INPUT_PATH = PROJ_PATH / 'data' / 'processed'
DEFAULT_OUTPUT_PATH = PROJ_PATH / 'data' / 'predictions'

MAX_FORECAST_HORIZON = 16

IGNORE_COLS = ['id', 'date', 'sales', 'sales_log1p']

DEFAULT_MODELS = ['xgb', 'lgbm']
DEFAULT_ENSEMBLE_WEIGHTS = {
    'xgb': 0.38,
    'lgbm': 0.62,
}

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    examples = """Examples:
      # Generate standard ensemble predictions:
      python src/predict.py

      # Generate predictions and apply holiday post-processing holiday rules:
      python src/predict.py --apply-holiday-rules
    """  
    parser = argparse.ArgumentParser(
        description='Generate store sales predictions using trained production models.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f'Input directory containing processed parquet files (default: {DEFAULT_INPUT_PATH}).',
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f'Directory containing trained .joblib models and metadata (default: {DEFAULT_MODEL_PATH}).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Output directory to save submission CSV (default: {DEFAULT_OUTPUT_PATH}).',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='submission.csv',
        help='Filename for the output predictions CSV (default: submission.csv).',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help='Model(s) to use for inference (default: xgb lgbm).',
    )
    parser.add_argument(
        '--apply-holiday-rules',
        action='store_true',
        help='Apply post-processing rules (zero out historically closed stores on Dec 25 and Jan 1).',
    )
    return parser.parse_args()


def load_metadata(models_dir: Path) -> dict | None:
    """Load model_metadata.json if available to extract features and ensemble weights."""
    metadata_path = models_dir / 'model_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def validate_test_dataset(df_test: pd.DataFrame, feature_cols: list[str]) -> None:
    """Validate forecast horizon length and feature schema integrity."""
    if df_test.empty:
        raise ValueError('Test dataset is empty.')

    min_date = pd.to_datetime(df_test['date'].min())
    max_date = pd.to_datetime(df_test['date'].max())
    horizon_days = (max_date - min_date).days + 1

    if horizon_days > MAX_FORECAST_HORIZON:
        raise ValueError(
            f'Test horizon [{min_date.date()} to {max_date.date()}] spans {horizon_days} days, '
            f'which exceeds the maximum supported forecast horizon of {MAX_FORECAST_HORIZON} days.'
        )

    missing_cols = [col for col in feature_cols if col not in df_test.columns]
    if missing_cols:
        raise ValueError(f'Test dataset is missing required feature columns: {missing_cols}')

    print(
        f'Validated test dataset: {len(df_test):,} rows | '
        f'Window: {min_date.date()} to {max_date.date()} ({horizon_days} days) | '
        f'Features: {len(feature_cols)}'
    )


def generate_predictions(
    X_test: pd.DataFrame,
    models_dir: Path,
    model_names: list[str],
    ensemble_weights: dict[str, float],
) -> pd.Series:
    """Load trained models, compute log-space predictions, blend, and invert via expm1."""
    preds_log1p = {}

    for model_name in model_names:
        model_path = models_dir / f'{model_name}_model.joblib'
        if not model_path.exists():
            raise FileNotFoundError(
                f'Trained model not found at {model_path}. '
                f'Please run `python src/train.py --full-train` first.'
            )

        print(f'Loading {model_name.upper()} model from {model_path.name}...')
        model = joblib.load(model_path)

        raw_preds = model.predict(X_test)
        preds_log1p[model_name] = pd.Series(
            np.clip(raw_preds, 0.0, None),
            index=X_test.index,
            name=f'{model_name}_preds_log1p',
        )

    total_weight = sum(ensemble_weights[m] for m in model_names)
    blended_log1p = sum(
        preds_log1p[m] * (ensemble_weights[m] / total_weight) for m in model_names
    )

    final_sales = np.expm1(blended_log1p).clip(lower=0.0)
    return final_sales


def apply_holiday_rules(
    df_test: pd.DataFrame,
    input_dir: Path,
) -> pd.Series:
    """Apply post-processing holiday closure business rules using historical sales patterns."""
    train_path = input_dir / 'train_features.parquet'
    if not train_path.exists():
        raise FileNotFoundError(
            f'Cannot apply holiday rules: {train_path} is required to check historical store closures.'
        )

    print('\n--- Applying Post-Processing Holiday Rules (Store Closures) ---')
    df_train = pd.read_parquet(
        train_path,
        columns=['date', 'store_nbr', 'sales_log1p'],
    )

    adjusted_sales = df_test['sales'].copy()
    test_dates = pd.to_datetime(df_test['date'])
    train_dates = pd.to_datetime(df_train['date'])

    # Zero out Dec 25 and Jan 1 for stores that are historically always closed on those dates
    for month, day, label in [(12, 25, 'Dec 25'), (1, 1, 'Jan 1')]:
        test_holiday_mask = (test_dates.dt.month == month) & (test_dates.dt.day == day)
        if test_holiday_mask.any():
            train_holiday = df_train[(train_dates.dt.month == month) & (train_dates.dt.day == day)]
            store_max_holiday_sales = train_holiday.groupby('store_nbr', observed=True)['sales_log1p'].max()
            closed_stores = store_max_holiday_sales[store_max_holiday_sales == 0].index

            rule_mask = test_holiday_mask & df_test['store_nbr'].isin(closed_stores)
            adjusted_sales.loc[rule_mask] = 0.0
            print(f'[Holiday Rule - {label}] Zeroed out {rule_mask.sum():,} rows across {len(closed_stores)} historically closed stores.')
        else:
            print(f'[Holiday Rule - {label}] No {label} dates present in test horizon (0 rows modified).')

    return adjusted_sales


def main() -> None:
    """Main execution entry point for generating test predictions."""
    args = parse_args()
    
    test_path = args.input_dir / 'test_features.parquet'
    if not test_path.exists():
        raise FileNotFoundError(f'Test features file not found at {test_path}.')

    print(f'Loading {test_path.name}...')
    df_test = pd.read_parquet(test_path)

    # Load metadata if available, otherwise fall back to defaults
    metadata = load_metadata(args.models_dir)
    if metadata is not None:
        feature_cols = metadata['features']['columns']
        ensemble_weights = metadata.get('ensemble_weights', DEFAULT_ENSEMBLE_WEIGHTS)
        print('Loaded feature schema and ensemble weights from model_metadata.json.')
    else:
        feature_cols = [col for col in df_test.columns if col not in IGNORE_COLS]
        ensemble_weights = DEFAULT_ENSEMBLE_WEIGHTS
        print('Warning: model_metadata.json not found. Using default columns and weights.')

    validate_test_dataset(df_test, feature_cols)
   
    X_test = df_test[feature_cols]

    df_test['sales'] = generate_predictions(
        X_test=X_test,
        models_dir=args.models_dir,
        model_names=args.models,
        ensemble_weights=ensemble_weights,
    )

    if args.apply_holiday_rules:
        df_test['sales'] = apply_holiday_rules(
            df_test=df_test,
            input_dir=args.input_dir,
        )

    # Post-prediction sanity check
    if df_test['sales'].isna().any():
        raise ValueError('Generated predictions contain NaN values!')
    if np.isinf(df_test['sales']).any():
        raise ValueError('Generated predictions contain Infinite values!')

    submission = df_test[['id', 'sales']].sort_values('id').reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    submission.to_csv(output_path, index=False)

    print(
        f'\nSaved {len(submission):,} predictions to {output_path} successfully!\n'
        f'Summary -> Min: {submission["sales"].min():.2f} | '
        f'Mean: {submission["sales"].mean():.2f} | '
        f'Max: {submission["sales"].max():.2f} | '
        f'Exact Zeros: {(submission["sales"] == 0).sum():,}'
    )


if __name__ == '__main__':
    main()