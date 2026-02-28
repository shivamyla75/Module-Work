"""
run_pipeline.py
===============
End-to-end pipeline runner for the HIGGS Boson Big Data ML Project.
Executes all 4 Jupyter notebooks sequentially using nbconvert,
with timing, logging, and error handling.

Usage:
    python scripts/run_pipeline.py [--skip-download] [--notebook N]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_run.log')
    ]
)
logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = Path(__file__).parent.parent / 'notebooks'

PIPELINE_STAGES = [
    {
        'id':   1,
        'name': 'Data Ingestion & Storage',
        'nb':   '1_data_ingestion.ipynb',
        'desc': 'Download HIGGS dataset, validate, convert to Parquet'
    },
    {
        'id':   2,
        'name': 'Feature Engineering & EDA',
        'nb':   '2_feature_engineering.ipynb',
        'desc': 'EDA, custom transformer, scaling, train/val/test split'
    },
    {
        'id':   3,
        'name': 'Model Training',
        'nb':   '3_model_training.ipynb',
        'desc': 'Train LR, RF, GBT, SVM via CrossValidator + sklearn baseline'
    },
    {
        'id':   4,
        'name': 'Model Evaluation & Scalability',
        'nb':   '4_evaluation.ipynb',
        'desc': 'ROC/PR curves, confusion matrices, bootstrap CI, scaling'
    }
]


def run_notebook(nb_path: Path, timeout: int = 7200) -> tuple[bool, float]:
    """
    Execute a Jupyter notebook using nbconvert.
    Returns (success: bool, elapsed_seconds: float).
    """
    output_path = nb_path.parent / f'executed_{nb_path.name}'
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        f'--ExecutePreprocessor.timeout={timeout}',
        '--inplace',
        '--output', str(output_path),
        str(nb_path)
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            logger.error(f'Notebook failed:\n{result.stderr[-3000:]}')
            return False, elapsed
        return True, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        logger.error(f'Notebook timed out after {timeout}s')
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f'Unexpected error: {e}')
        return False, elapsed


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║       HIGGS Boson Big Data ML Pipeline Runner            ║
║       Dataset: UCI HIGGS (~8GB, 11M rows, 29 cols)       ║
║       Framework: PySpark 3.5 + scikit-learn              ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description='Run HIGGS ML Pipeline')
    parser.add_argument('--notebook', type=int, default=None,
                        help='Run only notebook N (1-4). Default: run all.')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download (assumes HIGGS.csv already exists)')
    args = parser.parse_args()

    print_banner()

    # Filter stages if specific notebook requested
    stages = PIPELINE_STAGES
    if args.notebook:
        stages = [s for s in PIPELINE_STAGES if s['id'] == args.notebook]
        if not stages:
            logger.error(f'Notebook {args.notebook} not found. Valid: 1-4')
            sys.exit(1)

    pipeline_start = time.time()
    results = []

    for stage in stages:
        nb_path = NOTEBOOKS_DIR / stage['nb']
        logger.info(f'{"="*60}')
        logger.info(f'Stage {stage["id"]}/4: {stage["name"]}')
        logger.info(f'Notebook: {stage["nb"]}')
        logger.info(f'Description: {stage["desc"]}')
        logger.info(f'{"="*60}')

        if not nb_path.exists():
            logger.error(f'Notebook not found: {nb_path}')
            sys.exit(1)

        success, elapsed = run_notebook(nb_path)

        results.append({
            'stage': stage['id'],
            'name': stage['name'],
            'success': success,
            'elapsed_s': round(elapsed, 1),
            'elapsed_min': round(elapsed / 60, 2)
        })

        status = 'SUCCESS ✓' if success else 'FAILED ✗'
        logger.info(f'Stage {stage["id"]} {status} | Time: {elapsed/60:.1f} min')

        if not success:
            logger.error('Pipeline aborted due to stage failure.')
            _print_summary(results)
            sys.exit(1)

    total = time.time() - pipeline_start
    _print_summary(results, total)


def _print_summary(results: list, total_s: float = None):
    print('\n' + '='*60)
    print('PIPELINE EXECUTION SUMMARY')
    print('='*60)
    print(f'{"Stage":<6} {"Name":<35} {"Time (min)":<12} {"Status"}')
    print('-'*60)
    for r in results:
        status = '✓ OK' if r['success'] else '✗ FAILED'
        print(f'{r["stage"]:<6} {r["name"]:<35} {r["elapsed_min"]:<12} {status}')
    if total_s:
        print(f'\nTotal pipeline time: {total_s/60:.1f} minutes')
    print('='*60)
    print('\nOutputs saved to: data/samples/')
    print('Models saved to : data/models/')
    print('Open notebooks/  for detailed executed outputs.')


if __name__ == '__main__':
    main()
