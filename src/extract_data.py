import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from pyprojroot import here

PROJ_PATH = here()
SQL_PATH = PROJ_PATH / 'sql'
OUTPUT_PATH = PROJ_PATH / 'data' / 'interim'

load_dotenv(PROJ_PATH / '.env')

RUN_TARGETS = [
    'train',
    'test',
    'transactions',
]

MAX_GB_BILLED = 0.2

def get_query(target: str) -> str:
    if target == 'train':
        sql = (SQL_PATH / 'extract_base.sql').read_text()
        return sql.format(
            table_name='train', 
            target_col=', base.sales'
        )
    if target == 'test':
        sql = (SQL_PATH / 'extract_base.sql').read_text()
        return sql.format(
            table_name='test', 
            target_col=''
        )
    if target == 'transactions':
        return (SQL_PATH / 'extract_transactions.sql').read_text()
    raise ValueError(f"Unknown target: {target}")

def save_query_to_parquet(client: bigquery.Client, query: str, filename: str) -> None:
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=int(MAX_GB_BILLED*1024*1024*1024))
    query_job = client.query(query, job_config=safe_config)
    df = query_job.to_dataframe()
    df.to_parquet(OUTPUT_PATH / f'{filename}.parquet', index=False)
    print(f'Saved {filename}.parquet ({df.shape[0]:,} rows, {df.shape[1]} cols) successfully.')

def run_data_extraction(client: bigquery.Client, targets: list[str]) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(f'Creating SQL query for {target}...')
        query = get_query(target)   
        save_query_to_parquet(client, query, target)     

if __name__ == '__main__':
    project_id = os.getenv('GCP_PROJECT_ID')
    client = bigquery.Client(project=project_id)
    targets = sys.argv[1:] if len(sys.argv) > 1 else RUN_TARGETS
    run_data_extraction(client, targets)