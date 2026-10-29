import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from pyprojroot import here

PROJ_PATH = here()

DEFAULT_SQL_PATH = PROJ_PATH / 'sql'
DEFAULT_OUTPUT_PATH = PROJ_PATH / 'data' / 'interim'

DEFAULT_TARGETS = ['train', 'test', 'transactions']
DEFAULT_MAX_GB_BILLED = 0.2

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    examples = """Examples:
      # Extract all datasets using defaults:
      python src/extract_data.py

      # Dry-run to check estimated cost without billing:
      python src/extract_data.py --dry-run

      # Extract specific targets with custom limits:
      python src/extract_data.py --targets train test --max-gb-billed 0.5
    """  
    parser = argparse.ArgumentParser(
        description='Extract raw store sales datasets from BigQuery to Parquet.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--project-id',
        type=str,
        default=None,
        help='GCP Project ID (default: reads GCP_PROJECT_ID from .env).',
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=DEFAULT_TARGETS,
        default=DEFAULT_TARGETS,
        help='Dataset(s) to extract (default: train test transactions).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f'Directory to save parquet files (default: {DEFAULT_OUTPUT_PATH}).',
    )
    parser.add_argument(
        '--max-gb-billed',
        type=float,
        default=DEFAULT_MAX_GB_BILLED,
        help=f'Billing cap in Gigabytes per query (default: {DEFAULT_MAX_GB_BILLED} GB).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate queries and estimate bytes billed without executing or saving data.',
    )
    return parser.parse_args()

def get_query(target: str, sql_dir: Path = DEFAULT_SQL_PATH) -> str:
    """Build the SQL extraction query for the specified target table."""
    if target == 'train':
        sql = (sql_dir / 'extract_base.sql').read_text()
        return sql.format(table_name='train', target_col=', base.sales')
    if target == 'test':
        sql = (sql_dir / 'extract_base.sql').read_text()
        return sql.format(table_name='test', target_col='')
    if target == 'transactions':
        return (sql_dir / 'extract_transactions.sql').read_text()
    raise ValueError(f"Unknown target: {target}")

def save_query_to_parquet(
    client: bigquery.Client, 
    query: str, 
    filename: str,
    output_dir: Path,
    max_gb_billed: float,
    dry_run: bool = False
) -> None:
    """Execute BigQuery SQL job with a billing cap and save output to Parquet."""
    max_bytes = int(max_gb_billed * (1024**3))
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=max_bytes,
        dry_run=dry_run,
        use_query_cache=False if dry_run else True,
    )

    query_job = client.query(query, job_config=job_config)

    if dry_run:
        est_gb = query_job.total_bytes_processed / (1024**3)
        print(f'[DRY RUN] {filename}: Valid query. Estimated processing: {est_gb:.3f} GB.')
        return

    df = query_job.to_dataframe()
    output_file = output_dir / f'{filename}.parquet'
    df.to_parquet(output_file, index=False)
    print(f'Saved {output_file.name} ({df.shape[0]:,} rows, {df.shape[1]} cols) successfully.')

def main() -> None:
    """Main execution entry point for data extraction."""
    args = parse_args()

    load_dotenv(PROJ_PATH / '.env')
    project_id = args.project_id or os.getenv('GCP_PROJECT_ID')
    if not project_id:
        raise EnvironmentError(
            'GCP Project ID not found. Set GCP_PROJECT_ID in .env or pass --project-id.'
        )

    client = bigquery.Client(project=project_id)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for target in args.targets:
        print(f'Creating SQL query for {target}...')
        query = get_query(target)   
        save_query_to_parquet(
            client=client,
            query=query,
            filename=target,
            output_dir=args.output_dir,
            max_gb_billed=args.max_gb_billed,
            dry_run=args.dry_run,
        )     

if __name__ == '__main__':
    main()