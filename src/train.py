import argparse
from pathlib import Path
import db_dtypes
import joblib
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from pyprojroot import here
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

PROJ_PATH = here()

DEFAULT_INPUT_PATH = PROJ_PATH / 'data' / 'processed'
DEFAULT_MODEL_PATH = PROJ_PATH / 'models'

DEFAULT_VAL_START_DATE = '2017-07-31'
DEFAULT_VAL_DAYS = 16
MAX_FORECAST_HORIZON = 16

VAL_MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50

IGNORE_COLS = ['id', 'date', 'sales', 'sales_log1p']
TARGET_COL = 'sales_log1p'

DEFAULT_MODELS = ['xgb', 'lgbm']
ENSEMBLE_WEIGHTS = {
    'xgb': 0.38,
    'lgbm': 0.62,
}

XGB_BEST_PARAMS = {
    'tree_method': 'hist',
    'enable_categorical': True,
    'random_state': 42,
    'n_jobs': 6,
    'eval_metric': 'rmse',
    'n_estimators': 891, 
    'learning_rate': 0.0459538259120618, 
    'early_stopping_rounds': None,
    'max_depth': 9, 
    'min_child_weight': 3, 
    'subsample': 0.7275724643417782, 
    'colsample_bytree': 0.7735963189343423, 
    'reg_alpha': 0.0219410979090902, 
    'reg_lambda': 0.0037815840498335,
}

LGBM_BEST_PARAMS = {
    'objective': 'regression',
    'random_state': 42,
    'n_jobs': 6,
    'metric': 'rmse',
    'n_estimators': 1133,
    'learning_rate': 0.0379715140145848,
    'max_depth': -1, 
    'subsample_freq': 1,  
    'num_leaves': 235, 
    'min_child_samples': 11, 
    'subsample': 0.9877724173028624, 
    'colsample_bytree': 0.6804588926583752, 
    'reg_alpha': 0.8754015926417771, 
    'reg_lambda': 1.0165627777012725,
    'verbosity': -1,
}

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    examples = """Examples:
      # Run validation on the default 16-day window (2017-07-31 to 2017-08-15):
      python src/train.py

      # Backtest on a custom historical date and horizon:
      python src/train.py --val-start-date 2017-05-01 --val-days 16

      # Run default validation AND train final production models on 100%% of train data:
      python src/train.py --full-train
    """  
    parser = argparse.ArgumentParser(
        description='Validate and train XGBoost and LightGBM forecasting models.',
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
        help=f'Directory to save trained production models (default: {DEFAULT_MODEL_PATH}).',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help='Model(s) to evaluate/train (default: xgb lgbm).',
    )
    parser.add_argument(
        '--val-start-date',
        type=str,
        default=DEFAULT_VAL_START_DATE,
        help=f'Start date for the validation window (default: {DEFAULT_VAL_START_DATE}).',
    )
    parser.add_argument(
        '--val-days',
        type=int,
        default=DEFAULT_VAL_DAYS,
        help=f'Validation forecast horizon in days, <= 16 (default: {DEFAULT_VAL_DAYS}).',
    )
    parser.add_argument(
        '--full-train',
        action='store_true',
        help='Train models on 100%% of the training dataset and save them to --models-dir.',
    )
    return parser.parse_args()


def split_train_dataset(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str = TARGET_COL,
    val_start: str = DEFAULT_VAL_START_DATE,
    val_days: int = DEFAULT_VAL_DAYS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Splits data chronologically into train and validation sets."""
    start_dt = pd.to_datetime(val_start)
    end_dt = start_dt + pd.Timedelta(days=val_days-1)
    
    date_range_train = df['date'] < start_dt
    date_range_validation = (df['date'] >= start_dt) & (df['date'] <= end_dt)
    
    X_train = (df.loc[date_range_train, feature_cols]).reset_index(drop=True)
    y_train = (df.loc[date_range_train, target_col]).reset_index(drop=True)

    X_valid = (df.loc[date_range_validation, feature_cols]).reset_index(drop=True)
    y_valid = (df.loc[date_range_validation, target_col]).reset_index(drop=True)

    return X_train, y_train, X_valid, y_valid


def evaluate_model(
    model_name: str,    
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[float, pd.Series]:
    """Train a model on the train split with early stopping and evaluate on validation."""
    print(f'\n--- Evaluating {model_name.upper()} on validation split ---')

    if model_name == 'xgb':
        val_params = XGB_BEST_PARAMS | {
            'n_estimators': VAL_MAX_ESTIMATORS,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        }
        model = xgb.XGBRegressor(**val_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=100,
        )
        best_iter = model.best_iteration
    elif model_name == 'lgbm':
        val_params = LGBM_BEST_PARAMS | {
            'n_estimators': VAL_MAX_ESTIMATORS,
        }
        model = lgb.LGBMRegressor(**val_params)
        model.fit(
            X_train,
            y_train,
            eval_X=(X_train, X_valid),
            eval_y=(y_train, y_valid),
            eval_names=['train', 'val'],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        best_iter = model.best_iteration_
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    preds_valid = pd.Series(
        np.clip(model.predict(X_valid), 0.0, None),
        index=X_valid.index,
        name=f'{model_name}_preds_log1p',
    )

    val_rmsle = root_mean_squared_error(y_valid, preds_valid)
    
    print(
        f'[{model_name.upper()}] Validation RMSLE: {val_rmsle:.5f} (Best Iteration: {best_iter})'
    )

    return val_rmsle, preds_valid


def train_full_model(
    model_name: str,
    X_full: pd.DataFrame,
    y_full: pd.Series,
) -> xgb.XGBRegressor | lgb.LGBMRegressor:
    """Train a model on 100% of the training data using locked-in best parameters."""
    if model_name == 'xgb':
        print(
            f'\n--- Training full {model_name.upper()} (n_estimators={XGB_BEST_PARAMS["n_estimators"]}) ---'
        )
        model = xgb.XGBRegressor(**XGB_BEST_PARAMS)
        model.fit(X_full, y_full, verbose=100)
    elif model_name == 'lgbm':
        print(
            f'\n--- Training full {model_name.upper()} (n_estimators={LGBM_BEST_PARAMS["n_estimators"]}) ---'
        )
        model = lgb.LGBMRegressor(**LGBM_BEST_PARAMS)
        model.fit(
            X_full,
            y_full,
            callbacks=[lgb.log_evaluation(period=100)],
        )
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return model


def main() -> None:
    """Main execution entry point for model validation and full production training."""
    args = parse_args()

    if not (1 <= args.val_days <= MAX_FORECAST_HORIZON):
        raise ValueError(
            f'--val-days must be between 1 and {MAX_FORECAST_HORIZON} (got {args.val_days}).'
        )
    
    print('Loading processed train_features.parquet dataset...')
    df_train = pd.read_parquet(args.input_dir / 'train_features.parquet')

    # Validate date bounds dynamically against the loaded dataset
    val_start_dt = pd.to_datetime(args.val_start_date)
    val_end_dt = val_start_dt + pd.Timedelta(days=args.val_days - 1)
    max_dataset_dt = df_train['date'].max()
    if val_start_dt < pd.to_datetime('2013-07-01') or val_end_dt > max_dataset_dt:
        raise ValueError(
            f'Validation window [{val_start_dt.date()} to {val_end_dt.date()}] is out of bounds (allowed: 2013-07-01 to {max_dataset_dt.date()}).'
        )

    feature_cols = [col for col in df_train.columns if col not in IGNORE_COLS]

    # 1. Validation Step
    X_train, y_train, X_valid, y_valid = split_train_dataset(
        df = df_train, 
        feature_cols = feature_cols, 
        target_col = TARGET_COL,
        val_start = args.val_start_date,
        val_days=args.val_days
    )
    print(
        f'Validation Split: Train={len(X_train):,} rows (< {val_start_dt.date()}) | '
        f'Valid={len(X_valid):,} rows ({val_start_dt.date()} to {val_end_dt.date()})'
    )

    val_preds = {}
    for model_name in args.models:
        _, preds = evaluate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid
        )
        val_preds[model_name] = preds

    # Compute Ensemble RMSLE if both models were evaluated
    if len(val_preds) > 1:
        total_weight = sum(ENSEMBLE_WEIGHTS[m] for m in val_preds)
        blended_preds = sum(
            val_preds[m] * (ENSEMBLE_WEIGHTS[m] / total_weight) for m in val_preds
        )
        ensemble_rmsle = root_mean_squared_error(y_valid, blended_preds)
        print(f'\n>>> [ENSEMBLE] Validation RMSLE: {ensemble_rmsle:.5f} <<<')    

    # 2. Full Production Training Step (only if --full-train is passed)
    if args.full_train:
        X_full = df_train[feature_cols].reset_index(drop=True)
        y_full = df_train[TARGET_COL].reset_index(drop=True)
        print(f'\nStarting full production training on {len(X_full):,} rows...')

        args.models_dir.mkdir(parents=True, exist_ok=True)

        for model_name in args.models:
            full_model = train_full_model(
                model_name=model_name,
                X_full=X_full,
                y_full=y_full,
            )
            model_path = args.models_dir / f'{model_name}_model.joblib'
            joblib.dump(full_model, model_path)
            print(f'Saved {model_path.name} to {args.models_dir} successfully.')

        metadata = {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'validation_window': {
                'start_date': args.val_start_date,
                'days': args.val_days,
            },
            'validation_metrics_rmsle': {
                m: float(root_mean_squared_error(y_valid, val_preds[m])) for m in val_preds
            },
            'ensemble_weights': ENSEMBLE_WEIGHTS,
            'features': {
                'count': len(feature_cols),
                'columns': feature_cols,
            },
            'hyperparameters': {
                'xgb': XGB_BEST_PARAMS,
                'lgbm': LGBM_BEST_PARAMS,
            },
        }

        if len(val_preds) > 1:
            metadata['validation_metrics_rmsle']['ensemble'] = float(ensemble_rmsle)

        metadata_filename = args.models_dir / 'model_metadata.json'
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        print(f'Saved production metadata to {metadata_filename} successfully.')
    

if __name__ == '__main__':
    main()