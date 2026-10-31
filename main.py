import argparse
from pathlib import Path
import subprocess
import sys
import time
from pyprojroot import here

PROJ_PATH = here()
SRC_DIR = PROJ_PATH / 'src'

DEFAULT_INTERIM_DIR = PROJ_PATH / 'data' / 'interim'
DEFAULT_PROCESSED_DIR = PROJ_PATH / 'data' / 'processed'
DEFAULT_MODELS_DIR = PROJ_PATH / 'models'
DEFAULT_PREDICTIONS_DIR = PROJ_PATH / 'data' / 'predictions'

PIPELINE_STEPS = ['extract', 'features', 'train', 'predict']
EXTRACT_TARGETS = ['train', 'test', 'transactions']
DEFAULT_MODELS = ['xgb', 'lgbm']

DEFAULT_MAX_GB_BILLED = 0.2
DEFAULT_WARMUP_CUTOFF = '2013-03-01'
DEFAULT_VAL_START_DATE = '2017-07-31'
DEFAULT_VAL_DAYS = 16
DEFAULT_OUTPUT_NAME = 'submission.csv'


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for full end-to-end pipeline orchestration."""
    examples = """Examples:
      # 1. Run the entire end-to-end MLOps pipeline with default settings:
      python main.py

      # 2. Skip BigQuery extraction and run from feature engineering onwards:
      python main.py --steps features train predict

      # 3. Experiment with a 2016-06-01 warmup cutoff and save to a separate submission file:
      python main.py --steps features train predict --warmup-cutoff 2016-06-01 --output-name sub_2016_06.csv

      # 4. Run an isolated experiment in custom directories without overwriting default files:
      python main.py --steps features train predict --warmup-cutoff 2016-01-01 \\
          --processed-dir data/processed_2016 --models-dir models/exp_2016 --output-name sub_2016_01.csv

      # 5. Run only validation (without full training) on a holiday window:
      python main.py --steps train --val-only --val-start-date 2016-12-19 --val-days 16
    """
    parser = argparse.ArgumentParser(
        description='End-to-End Store Sales Forecasting MLOps Pipeline Orchestrator.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Group 1: Orchestrator & Directory Routing
    orch_group = parser.add_argument_group('Pipeline Control & Directories')
    orch_group.add_argument(
        '--steps',
        nargs='+',
        choices=PIPELINE_STEPS,
        default=PIPELINE_STEPS,
        help='Pipeline steps to execute in sequence (default: extract features train predict).',
    )
    orch_group.add_argument(
        '--interim-dir',
        type=Path,
        default=DEFAULT_INTERIM_DIR,
        help=f'Directory for interim parquet files: extract output -> features input (default: {DEFAULT_INTERIM_DIR}).',
    )
    orch_group.add_argument(
        '--processed-dir',
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f'Directory for processed feature parquets: features output -> train/predict input (default: {DEFAULT_PROCESSED_DIR}).',
    )
    orch_group.add_argument(
        '--models-dir',
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f'Directory for trained models & metadata: train output -> predict input (default: {DEFAULT_MODELS_DIR}).',
    )
    orch_group.add_argument(
        '--predictions-dir',
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f'Directory to save final predictions CSV (default: {DEFAULT_PREDICTIONS_DIR}).',
    )

    # Group 2: Data Extraction (src/extract_data.py)
    extract_group = parser.add_argument_group('Step 1: Data Extraction (extract_data.py)')
    extract_group.add_argument(
        '--project-id',
        type=str,
        default=None,
        help='GCP Project ID (default: reads GCP_PROJECT_ID from .env).',
    )
    extract_group.add_argument(
        '--targets',
        nargs='+',
        choices=EXTRACT_TARGETS,
        default=EXTRACT_TARGETS,
        help='Dataset(s) to extract from BigQuery (default: train test transactions).',
    )
    extract_group.add_argument(
        '--max-gb-billed',
        type=float,
        default=DEFAULT_MAX_GB_BILLED,
        help=f'Billing cap in Gigabytes per query (default: {DEFAULT_MAX_GB_BILLED} GB).',
    )
    extract_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate BigQuery queries and estimate cost without executing or saving data.',
    )

    # Group 3: Feature Engineering (src/build_features.py)
    feat_group = parser.add_argument_group('Step 2: Feature Engineering (build_features.py)')
    feat_group.add_argument(
        '--warmup-cutoff',
        type=str,
        default=DEFAULT_WARMUP_CUTOFF,
        help=f'Earliest date for training samples after lag warm-up (default: {DEFAULT_WARMUP_CUTOFF}).',
    )

    # Group 4: Model Training & Inference (src/train.py & src/predict.py)
    model_group = parser.add_argument_group('Step 3 & 4: Training & Inference (train.py & predict.py)')
    model_group.add_argument(
        '--models',
        nargs='+',
        choices=DEFAULT_MODELS,
        default=DEFAULT_MODELS,
        help='Model(s) to validate, train, and use for inference (default: xgb lgbm).',
    )
    model_group.add_argument(
        '--val-start-date',
        type=str,
        default=DEFAULT_VAL_START_DATE,
        help=f'Start date for the validation window in train.py (default: {DEFAULT_VAL_START_DATE}).',
    )
    model_group.add_argument(
        '--val-days',
        type=int,
        default=DEFAULT_VAL_DAYS,
        help=f'Validation forecast horizon in days, <= 16 (default: {DEFAULT_VAL_DAYS}).',
    )
    model_group.add_argument(
        '--val-only',
        action='store_true',
        help='Run only validation in train.py without training/saving full production models.',
    )
    model_group.add_argument(
        '--no-holiday-rules',
        action='store_true',
        help='Disable post-processing holiday closure rules (enabled by default in the pipeline).',
    )
    model_group.add_argument(
        '--output-name',
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=f'Filename for the output predictions CSV in predict.py (default: {DEFAULT_OUTPUT_NAME}).',
    )

    return parser.parse_args()


def run_step(step_name: str, cmd: list[str]) -> float:
    """Execute a pipeline step via subprocess and return elapsed time in seconds."""
    cmd_str = ' '.join(str(part) for part in cmd)
    print(f'\n{"=" * 85}')
    print(f'[PIPELINE STEP: {step_name}]')
    print(f'Command: {cmd_str}')
    print(f'{"=" * 85}\n')

    start_time = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f'\n[ERROR] Pipeline failed at "{step_name}" with exit code {exc.returncode}.')
        sys.exit(exc.returncode)

    elapsed = time.perf_counter() - start_time
    print(f'\n[COMPLETED: {step_name} in {elapsed:.2f}s]')
    return elapsed


def main() -> None:
    """Main entry point to orchestrate the forecasting pipeline."""
    args = parse_args()
    python_exec = sys.executable
    apply_holiday_rules = not args.no_holiday_rules

    step_timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    # Step 1: Data Extraction (src/extract_data.py)
    if 'extract' in args.steps:
        extract_cmd = [
            python_exec,
            str(SRC_DIR / 'extract_data.py'),
            '--targets',
            *args.targets,
            '--output-dir',
            str(args.interim_dir),
            '--max-gb-billed',
            str(args.max_gb_billed),
        ]
        if args.project_id is not None:
            extract_cmd.extend(['--project-id', args.project_id])
        if args.dry_run:
            extract_cmd.append('--dry-run')

        step_timings['1. Extract Data'] = run_step('1. DATA EXTRACTION', extract_cmd)

    # Step 2: Feature Engineering (src/build_features.py)
    if 'features' in args.steps:
        features_cmd = [
            python_exec,
            str(SRC_DIR / 'build_features.py'),
            '--input-dir',
            str(args.interim_dir),
            '--output-dir',
            str(args.processed_dir),
            '--warmup-cutoff',
            args.warmup_cutoff,
        ]

        step_timings['2. Build Features'] = run_step('2. FEATURE ENGINEERING', features_cmd)    

    # Step 3: Model Validation & Training (src/train.py)
    if 'train' in args.steps:
        train_cmd = [
            python_exec,
            str(SRC_DIR / 'train.py'),
            '--input-dir',
            str(args.processed_dir),
            '--models-dir',
            str(args.models_dir),
            '--models',
            *args.models,
            '--val-start-date',
            args.val_start_date,
            '--val-days',
            str(args.val_days),
        ]
        if not args.val_only:
            train_cmd.append('--full-train')
        if apply_holiday_rules:
            train_cmd.append('--apply-holiday-rules')

        step_timings['3. Train Models'] = run_step('3. MODEL VALIDATION & TRAINING', train_cmd)   

    # Step 4: Model Inference (src/predict.py)
    if 'predict' in args.steps:
        predict_cmd = [
            python_exec,
            str(SRC_DIR / 'predict.py'),
            '--input-dir',
            str(args.processed_dir),
            '--models-dir',
            str(args.models_dir),
            '--output-dir',
            str(args.predictions_dir),
            '--output-name',
            args.output_name,
            '--models',
            *args.models,
        ]
        if apply_holiday_rules:
            predict_cmd.append('--apply-holiday-rules')

        step_timings['4. Predict Sales'] = run_step('4. MODEL INFERENCE', predict_cmd)  

    total_elapsed = time.perf_counter() - pipeline_start

    print(f'\n{"=" * 85}')
    print('PIPELINE EXECUTION SUMMARY')
    print(f'{"-" * 85}')
    for step_label, duration in step_timings.items():
        print(f'  {step_label.ljust(22)} : {duration:>8.2f}s')
    print(f'{"-" * 85}')
    print(f'  {"Total Execution Time".ljust(22)} : {total_elapsed:>8.2f}s')
    print(f'{"=" * 85}')       


if __name__ == '__main__':
    main()