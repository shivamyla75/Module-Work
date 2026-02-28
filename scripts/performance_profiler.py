"""
performance_profiler.py
=======================
Profiles Spark job performance metrics:
- Stage timings from Spark History Server API
- Memory usage per executor
- Shuffle read/write stats
- GC time analysis
- Bottleneck identification (I/O vs compute vs shuffle)

Usage:
    python scripts/performance_profiler.py --app-id <spark-app-id>
    python scripts/performance_profiler.py --live  (connects to running Spark UI)
"""

import argparse
import json
import time
import sys
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

SPARK_UI_BASE = 'http://localhost:4040'
HISTORY_BASE  = 'http://localhost:18080'


class SparkProfiler:
    """
    Connects to Spark REST API and extracts performance metrics.
    Compatible with Spark 3.x REST API.
    """

    def __init__(self, base_url: str = SPARK_UI_BASE, app_id: str = None):
        self.base_url = base_url.rstrip('/')
        self.app_id   = app_id
        self.session  = requests.Session()
        self.session.timeout = 10

    def _get(self, endpoint: str) -> dict | list:
        url = f'{self.base_url}/api/v1{endpoint}'
        try:
            r = self.session.get(url)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            logger.error(f'Cannot connect to Spark UI at {self.base_url}')
            return {}
        except requests.HTTPError as e:
            logger.error(f'HTTP error: {e}')
            return {}

    def get_applications(self) -> list:
        return self._get('/applications') or []

    def get_jobs(self, app_id: str) -> list:
        return self._get(f'/applications/{app_id}/jobs') or []

    def get_stages(self, app_id: str) -> list:
        return self._get(f'/applications/{app_id}/stages') or []

    def get_executors(self, app_id: str) -> list:
        return self._get(f'/applications/{app_id}/executors') or []

    def profile_application(self, app_id: str) -> dict:
        """Full profile of a Spark application."""
        logger.info(f'Profiling application: {app_id}')

        jobs     = self.get_jobs(app_id)
        stages   = self.get_stages(app_id)
        executors= self.get_executors(app_id)

        if not stages:
            logger.warning('No stages found. Is the application running/complete?')
            return {}

        # ---- Stage Analysis ----
        stage_metrics = []
        for s in stages:
            stage_metrics.append({
                'stage_id':          s.get('stageId'),
                'name':              s.get('name', '')[:50],
                'status':            s.get('status'),
                'num_tasks':         s.get('numTasks', 0),
                'duration_ms':       s.get('executorRunTime', 0),
                'input_bytes':       s.get('inputBytes', 0),
                'output_bytes':      s.get('outputBytes', 0),
                'shuffle_read_bytes':s.get('shuffleReadBytes', 0),
                'shuffle_write_bytes':s.get('shuffleWriteBytes', 0),
                'gc_time_ms':        s.get('jvmGcTime', 0),
                'executor_cpu_time_ms': s.get('executorCpuTime', 0) / 1e6,
            })

        stage_df = pd.DataFrame(stage_metrics)

        # ---- Executor Analysis ----
        exec_metrics = []
        for e in executors:
            exec_metrics.append({
                'executor_id':    e.get('id'),
                'host':           e.get('hostPort', ''),
                'cores':          e.get('totalCores', 0),
                'max_memory_mb':  e.get('maxMemory', 0) / 1024**2,
                'used_memory_mb': e.get('memoryUsed', 0) / 1024**2,
                'tasks_active':   e.get('activeTasks', 0),
                'tasks_complete': e.get('completedTasks', 0),
                'tasks_failed':   e.get('failedTasks', 0),
                'gc_time_ms':     e.get('totalGCTime', 0),
            })

        exec_df = pd.DataFrame(exec_metrics) if exec_metrics else pd.DataFrame()

        # ---- Bottleneck Identification ----
        bottleneck = self._identify_bottleneck(stage_df)

        return {
            'stages':     stage_df,
            'executors':  exec_df,
            'bottleneck': bottleneck,
            'summary': {
                'total_stages':       len(stage_df),
                'total_tasks':        stage_df['num_tasks'].sum(),
                'total_runtime_min':  stage_df['duration_ms'].sum() / 60000,
                'total_shuffle_gb':   (stage_df['shuffle_read_bytes'].sum() +
                                       stage_df['shuffle_write_bytes'].sum()) / 1e9,
                'total_gc_time_min':  stage_df['gc_time_ms'].sum() / 60000,
                'app_id':             app_id
            }
        }

    def _identify_bottleneck(self, stage_df: pd.DataFrame) -> str:
        """Heuristic bottleneck identification."""
        if stage_df.empty:
            return 'Unknown'

        total_runtime   = stage_df['duration_ms'].sum()
        total_shuffle   = stage_df['shuffle_read_bytes'].sum() + stage_df['shuffle_write_bytes'].sum()
        total_gc        = stage_df['gc_time_ms'].sum()
        total_input_io  = stage_df['input_bytes'].sum()

        gc_ratio      = total_gc / max(total_runtime, 1)
        shuffle_gb    = total_shuffle / 1e9
        input_gb      = total_input_io / 1e9

        if gc_ratio > 0.2:
            return f'GC_PRESSURE ({gc_ratio*100:.1f}% of runtime in GC — increase executor memory)'
        elif shuffle_gb > 50:
            return f'SHUFFLE_HEAVY ({shuffle_gb:.1f}GB shuffle — increase shuffle partitions or use broadcast joins)'
        elif input_gb > 100 and total_runtime > 600_000:
            return f'IO_BOUND ({input_gb:.1f}GB input — consider caching hot DataFrames)'
        else:
            return f'COMPUTE_BOUND (no major I/O or GC issues identified)'


def generate_profile_report(profile: dict, output_dir: Path):
    """Generate visual performance report."""
    output_dir.mkdir(exist_ok=True)

    stage_df = profile.get('stages', pd.DataFrame())
    exec_df  = profile.get('executors', pd.DataFrame())
    summary  = profile.get('summary', {})

    if stage_df.empty:
        logger.warning('No data to plot.')
        return

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Stage duration bar
    ax1 = fig.add_subplot(gs[0, :])
    top_stages = stage_df.nlargest(15, 'duration_ms')
    ax1.barh(top_stages['name'], top_stages['duration_ms'] / 1000,
             color='#2196F3', alpha=0.85)
    ax1.set_xlabel('Duration (s)')
    ax1.set_title('Top 15 Slowest Stages')
    ax1.grid(axis='x', alpha=0.3)

    # 2. Shuffle read vs write
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(stage_df['shuffle_read_bytes']/1e6,
                stage_df['shuffle_write_bytes']/1e6,
                alpha=0.6, color='#FF5722')
    ax2.set_xlabel('Shuffle Read (MB)')
    ax2.set_ylabel('Shuffle Write (MB)')
    ax2.set_title('Shuffle Read vs Write per Stage')
    ax2.grid(alpha=0.3)

    # 3. GC time vs runtime
    ax3 = fig.add_subplot(gs[1, 1])
    if not stage_df['gc_time_ms'].empty:
        ax3.bar(stage_df['stage_id'], stage_df['gc_time_ms']/1000,
                color='#FF9800', alpha=0.8)
        ax3.set_xlabel('Stage ID')
        ax3.set_ylabel('GC Time (s)')
        ax3.set_title('GC Time per Stage')
        ax3.grid(axis='y', alpha=0.3)

    # 4. Executor memory usage
    ax4 = fig.add_subplot(gs[2, 0])
    if not exec_df.empty:
        x = range(len(exec_df))
        ax4.bar(x, exec_df['max_memory_mb']/1024, label='Max Memory (GB)', alpha=0.6, color='#9E9E9E')
        ax4.bar(x, exec_df['used_memory_mb']/1024, label='Used Memory (GB)', alpha=0.8, color='#2196F3')
        ax4.set_xticks(x)
        ax4.set_xticklabels(exec_df['executor_id'], rotation=45)
        ax4.set_ylabel('Memory (GB)')
        ax4.set_title('Executor Memory Usage')
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)

    # 5. Summary KPIs
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    kpis = [
        ['Total Stages',      str(summary.get('total_stages', '-'))],
        ['Total Tasks',       f"{summary.get('total_tasks', 0):,}"],
        ['Total Runtime',     f"{summary.get('total_runtime_min', 0):.1f} min"],
        ['Total Shuffle',     f"{summary.get('total_shuffle_gb', 0):.2f} GB"],
        ['GC Time',           f"{summary.get('total_gc_time_min', 0):.2f} min"],
        ['Bottleneck',        profile.get('bottleneck', '-')[:40]]
    ]
    table = ax5.table(cellText=kpis, colLabels=['Metric', 'Value'],
                      cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(True)
    ax5.set_title('Performance Summary', fontweight='bold')

    fig.suptitle(f'Spark Performance Profile — App: {summary.get("app_id", "N/A")}',
                 fontsize=14, fontweight='bold')

    out_path = output_dir / 'spark_performance_profile.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    logger.info(f'Profile saved: {out_path}')

    # Save CSVs for Tableau
    stage_df.to_csv(output_dir / 'stage_metrics.csv', index=False)
    if not exec_df.empty:
        exec_df.to_csv(output_dir / 'executor_metrics.csv', index=False)

    plt.show()


def generate_mock_profile(output_dir: Path):
    """
    Generate mock profile data for demonstration when no live Spark app exists.
    Useful for testing/Tableau dashboard development.
    """
    logger.info('Generating mock performance profile...')

    import numpy as np
    rng = np.random.default_rng(42)

    stage_names = [
        'csv read', 'fillna/median', 'repartition', 'VectorAssembler',
        'StandardScaler.fit', 'StandardScaler.transform', 'PCA.fit',
        'randomSplit', 'LR.fit CV fold 1', 'LR.fit CV fold 2',
        'RF.fit CV fold 1', 'RF.fit CV fold 2', 'GBT.fit CV fold 1',
        'GBT.fit CV fold 2', 'SVM.fit', 'evaluator.AUC', 'parquet write'
    ]

    n = len(stage_names)
    stage_df = pd.DataFrame({
        'stage_id':              range(n),
        'name':                  stage_names,
        'status':                ['COMPLETE'] * n,
        'num_tasks':             rng.integers(50, 400, n),
        'duration_ms':           rng.integers(5000, 900000, n),
        'input_bytes':           rng.integers(0, int(8e9), n),
        'output_bytes':          rng.integers(0, int(2e9), n),
        'shuffle_read_bytes':    rng.integers(0, int(3e9), n),
        'shuffle_write_bytes':   rng.integers(0, int(3e9), n),
        'gc_time_ms':            rng.integers(100, 30000, n),
        'executor_cpu_time_ms':  rng.integers(2000, 800000, n),
    })

    exec_df = pd.DataFrame({
        'executor_id':    ['driver', '1', '2', '3', '4'],
        'host':           ['localhost:36000'] * 5,
        'cores':          [0, 4, 4, 4, 4],
        'max_memory_mb':  [8192, 6144, 6144, 6144, 6144],
        'used_memory_mb': rng.integers(2048, 5500, 5),
        'tasks_active':   [0, 0, 0, 0, 0],
        'tasks_complete': rng.integers(200, 2000, 5),
        'tasks_failed':   rng.integers(0, 5, 5),
        'gc_time_ms':     rng.integers(5000, 60000, 5),
    })

    profile = {
        'stages':     stage_df,
        'executors':  exec_df,
        'bottleneck': 'SHUFFLE_HEAVY (estimated ~12GB shuffle across CV folds — consider broadcast)',
        'summary': {
            'total_stages':       n,
            'total_tasks':        int(stage_df['num_tasks'].sum()),
            'total_runtime_min':  stage_df['duration_ms'].sum() / 60000,
            'total_shuffle_gb':   (stage_df['shuffle_read_bytes'].sum() +
                                   stage_df['shuffle_write_bytes'].sum()) / 1e9,
            'total_gc_time_min':  stage_df['gc_time_ms'].sum() / 60000,
            'app_id':             'HIGGS-Mock-Profile'
        }
    }

    generate_profile_report(profile, output_dir)
    return profile


def main():
    parser = argparse.ArgumentParser(description='Spark Performance Profiler')
    parser.add_argument('--app-id',  type=str, default=None,
                        help='Spark application ID to profile')
    parser.add_argument('--live',    action='store_true',
                        help='Connect to running Spark UI at localhost:4040')
    parser.add_argument('--mock',    action='store_true',
                        help='Generate mock profile for testing')
    parser.add_argument('--output',  type=str, default='data/samples',
                        help='Output directory for profile reports')
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.mock:
        generate_mock_profile(output_dir)
        return

    profiler = SparkProfiler(
        base_url=SPARK_UI_BASE if args.live else HISTORY_BASE
    )

    if args.live:
        # Discover running app
        apps = profiler.get_applications()
        if not apps:
            logger.warning('No live application found. Using mock profile.')
            generate_mock_profile(output_dir)
            return
        app_id = apps[0]['id']
    elif args.app_id:
        app_id = args.app_id
    else:
        logger.info('No app specified. Generating mock profile.')
        generate_mock_profile(output_dir)
        return

    profile = profiler.profile_application(app_id)
    if profile:
        generate_profile_report(profile, output_dir)
        print(f"\nBottleneck: {profile['bottleneck']}")
    else:
        logger.error('Failed to retrieve profile. Check Spark UI is accessible.')


if __name__ == '__main__':
    main()
