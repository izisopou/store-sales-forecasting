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

VAL_MAX_ESTIMATORS = 2500
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

      # Backtest on a holiday window and apply post-processing holiday rules:
      python src/train.py --val-start-date 2016-12-20 --val-days 16 --apply-holiday-rules

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
    parser.add_argument(
        '--apply-holiday-rules',
        action='store_true',
        help='Apply post-processing holiday rules to validation predictions (zero out closed stores on Dec 25 and Jan 1).',
    )
    return parser.parse_args()


def split_train_dataset(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    target_col: str = TARGET_COL,
    val_start: str = DEFAULT_VAL_START_DATE,
    val_days: int = DEFAULT_VAL_DAYS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Splits data chronologically into train and validation sets."""
    start_dt = pd.to_datetime(val_start)
    end_dt = start_dt + pd.Timedelta(days=val_days-1)
    
    date_range_train = df['date'] < start_dt
    date_range_validation = (df['date'] >= start_dt) & (df['date'] <= end_dt)

    df_train_split = df.loc[date_range_train].reset_index(drop=True)
    df_valid_split = df.loc[date_range_validation].reset_index(drop=True)
    
    X_train = df_train_split[feature_cols]
    y_train = df_train_split[target_col]

    X_valid = df_valid_split[feature_cols]
    y_valid = df_valid_split[target_col]

    return X_train, y_train, X_valid, y_valid, df_train_split, df_valid_split


def get_holiday_rules_mask(
    df_train_split: pd.DataFrame,
    df_valid_split: pd.DataFrame,
) -> pd.Series:
    """Return a boolean mask of validation rows that should be zeroed out on Dec 25 and Jan 1."""
    print('\n--- Computing Post-Processing Holiday Rules (Store Closures) ---')
    valid_dates = pd.to_datetime(df_valid_split['date'])
    train_dates = pd.to_datetime(df_train_split['date'])
    zero_mask = pd.Series(False, index=df_valid_split.index)

    # Zero out Dec 25 and Jan 1 for stores that are historically always closed on those dates
    for month, day, label in [(12, 25, 'Dec 25'), (1, 1, 'Jan 1')]:
        val_holiday_mask = (valid_dates.dt.month == month) & (valid_dates.dt.day == day)
        if val_holiday_mask.any():
            train_holiday = df_train_split[(train_dates.dt.month == month) & (train_dates.dt.day == day)]
            store_max = train_holiday.groupby('store_nbr', observed=True)[TARGET_COL].max()
            closed_stores = store_max[store_max == 0].index

            rule1_mask = val_holiday_mask & df_valid_split['store_nbr'].isin(closed_stores)
            zero_mask |= rule1_mask
            print(f'[{label}] Zeroed out {rule1_mask.sum():,} rows across {len(closed_stores)} historically closed stores.')
        else:
            print(f'[{label}] No {label} dates present in validation horizon (0 rows modified).')

    return zero_mask


def evaluate_model(
    model_name: str,    
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[float, pd.Series, int]:
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

    return val_rmsle, preds_valid, int(best_iter)


def train_full_model(
    model_name: str,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    n_estimators: int | None = None,
) -> tuple[xgb.XGBRegressor | lgb.LGBMRegressor, dict]:
    """Train a model on 100% of the training data using best parameters and dynamic n_estimators."""
    if model_name == 'xgb':
        params = XGB_BEST_PARAMS.copy()
        if n_estimators is not None:
            params['n_estimators'] = int(n_estimators)
        print(
            f'\n--- Training full {model_name.upper()} (n_estimators={params["n_estimators"]}) ---'
        )
        model = xgb.XGBRegressor(**params)
        model.fit(X_full, y_full, verbose=100)
    elif model_name == 'lgbm':
        params = LGBM_BEST_PARAMS.copy()
        if n_estimators is not None:
            params['n_estimators'] = int(n_estimators)
        print(
            f'\n--- Training full {model_name.upper()} (n_estimators={params["n_estimators"]}) ---'
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_full,
            y_full,
            callbacks=[lgb.log_evaluation(period=100)],
        )
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return model, params


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
    min_dataset_dt = pd.to_datetime(df_train['date'].min())
    max_dataset_dt = pd.to_datetime(df_train['date'].max())
    if val_start_dt <= min_dataset_dt or val_end_dt > max_dataset_dt:
        raise ValueError(
            f'Validation window [{val_start_dt.date()} to {val_end_dt.date()}] is out of bounds '
            f'for loaded dataset range [{min_dataset_dt.date()} to {max_dataset_dt.date()}].'
        )

    feature_cols = [col for col in df_train.columns if col not in IGNORE_COLS]

    # 1. Validation Step
    X_train, y_train, X_valid, y_valid, df_train_split, df_valid_split = split_train_dataset(
        df = df_train, 
        feature_cols = feature_cols, 
        target_col = TARGET_COL,
        val_start = args.val_start_date,
        val_days=args.val_days,
    )
    print(
        f'Validation Split: Train={len(X_train):,} rows (< {val_start_dt.date()}) | '
        f'Valid={len(X_valid):,} rows ({val_start_dt.date()} to {val_end_dt.date()})'
    )

    val_preds = {}
    best_iters = {}
    for model_name in args.models:
        _, preds, best_iter = evaluate_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        val_preds[model_name] = preds
        best_iters[model_name] = best_iter

    # Compute Raw Ensemble RMSLE if both models were evaluated
    if len(val_preds) > 1:
        total_weight = sum(ENSEMBLE_WEIGHTS[m] for m in val_preds)
        blended_preds = sum(
            val_preds[m] * (ENSEMBLE_WEIGHTS[m] / total_weight) for m in val_preds
        )
        ensemble_rmsle = root_mean_squared_error(y_valid, blended_preds)
        print(f'\n>>> [ENSEMBLE - RAW] Validation RMSLE: {ensemble_rmsle:.5f} <<<')  

    # Optional Post-Processing Holiday Rules Evaluation
    post_rule_metrics = {}
    if args.apply_holiday_rules:
        zero_mask = get_holiday_rules_mask(
            df_train_split=df_train_split,
            df_valid_split=df_valid_split,
        )
        for model_name, preds in val_preds.items():
            adj_preds = preds.copy()
            adj_preds.loc[zero_mask] = 0.0
            adj_rmsle = root_mean_squared_error(y_valid, adj_preds)
            post_rule_metrics[model_name] = float(adj_rmsle)
            print(f'[{model_name.upper()} - HOLIDAY RULES] Validation RMSLE: {adj_rmsle:.5f}')

        if len(val_preds) > 1:
            adj_blended = blended_preds.copy()
            adj_blended.loc[zero_mask] = 0.0
            adj_ensemble_rmsle = root_mean_squared_error(y_valid, adj_blended)
            post_rule_metrics['ensemble'] = float(adj_ensemble_rmsle)
            print(f'>>> [ENSEMBLE - HOLIDAY RULES] Validation RMSLE: {adj_ensemble_rmsle:.5f} <<<')  

    # 2. Full Production Training Step (only if --full-train is passed)
    if args.full_train:
        is_default_val = (
            args.val_start_date == DEFAULT_VAL_START_DATE
            and args.val_days == DEFAULT_VAL_DAYS
        )
        if not is_default_val:
            print(
                f'\nCustom validation window was used ({args.val_start_date}). '
                f'Running reference validation on [{DEFAULT_VAL_START_DATE}, {DEFAULT_VAL_DAYS} days] '
                f'to determine optimal n_estimators for full training...'
            )
            X_tr_ref, y_tr_ref, X_val_ref, y_val_ref, _, _ = split_train_dataset(
                df=df_train,
                feature_cols=feature_cols,
                target_col=TARGET_COL,
                val_start=DEFAULT_VAL_START_DATE,
                val_days=DEFAULT_VAL_DAYS,
            )
            for model_name in args.models:
                _, _, ref_best_iter = evaluate_model(
                    model_name=model_name,
                    X_train=X_tr_ref,
                    y_train=y_tr_ref,
                    X_valid=X_val_ref,
                    y_valid=y_val_ref,
                )
                best_iters[model_name] = ref_best_iter

        X_full = df_train[feature_cols].reset_index(drop=True)
        y_full = df_train[TARGET_COL].reset_index(drop=True)
        print(f'\nStarting full production training on {len(X_full):,} rows...')

        args.models_dir.mkdir(parents=True, exist_ok=True)

        used_hyperparams = {}
        for model_name in args.models:
            full_model, model_params = train_full_model(
                model_name=model_name,
                X_full=X_full,
                y_full=y_full,
                n_estimators=best_iters[model_name],
            )
            used_hyperparams[model_name] = model_params
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
            'hyperparameters': used_hyperparams,
        }

        if len(val_preds) > 1:
            metadata['validation_metrics_rmsle']['ensemble'] = float(ensemble_rmsle)

        if post_rule_metrics:
            metadata['validation_metrics_rmsle_holiday_rules'] = post_rule_metrics

        metadata_filename = args.models_dir / 'model_metadata.json'
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        print(f'Saved production metadata to {metadata_filename} successfully.')
    

if __name__ == '__main__':
    main()