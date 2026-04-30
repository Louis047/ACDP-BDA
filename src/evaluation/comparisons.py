"""
Compare ACADP vs baselines with comprehensive metrics.
Supports three-way comparison (ACADP vs Uniform vs Random Blocking)
and multi-ε sweep for publication-quality results.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List, Set
import logging

from .utility_metrics import compute_all_utility_metrics
from .baselines import baseline_uniform_privatize, baseline_random_blocking_privatize
from ..dp.privatize import DatasetPrivatizer

logger = logging.getLogger(__name__)


def compare_three_way(
    original_df: pd.DataFrame,
    acadp_df: pd.DataFrame,
    uniform_df: pd.DataFrame,
    random_df: pd.DataFrame,
    columns: Optional[list] = None
) -> Dict:
    """
    Three-way comparison: ACADP vs Uniform Baseline vs Random Blocking.

    Args:
        original_df: Original dataset
        acadp_df: ACADP-privatized dataset
        uniform_df: Uniform baseline privatized dataset
        random_df: Random blocking privatized dataset
        columns: Columns to compare

    Returns:
        Dict with comprehensive comparison metrics
    """
    logger.info("Computing ACADP metrics...")
    acadp_metrics = compute_all_utility_metrics(original_df, acadp_df, columns)

    logger.info("Computing uniform baseline metrics...")
    uniform_metrics = compute_all_utility_metrics(original_df, uniform_df, columns)

    logger.info("Computing random blocking metrics...")
    random_metrics = compute_all_utility_metrics(original_df, random_df, columns)

    # Compute improvements
    def _improvement_pct(baseline_val, acadp_val):
        """Positive = ACADP is better (lower error)."""
        if baseline_val == 0:
            return 0.0
        return (baseline_val - acadp_val) / baseline_val * 100

    improvements_vs_uniform = {}
    improvements_vs_random = {}

    for metric_key in ['mean_mse', 'mean_mae', 'mean_relative_error',
                       'correlation_error', 'mean_kl_divergence',
                       'mean_js_divergence', 'mean_wasserstein_distance',
                       'mean_ks_statistic', 'mean_query_error']:
        if metric_key in acadp_metrics and metric_key in uniform_metrics:
            improvements_vs_uniform[metric_key] = _improvement_pct(
                uniform_metrics[metric_key], acadp_metrics[metric_key]
            )
        if metric_key in acadp_metrics and metric_key in random_metrics:
            improvements_vs_random[metric_key] = _improvement_pct(
                random_metrics[metric_key], acadp_metrics[metric_key]
            )

    comparison = {
        'acadp_metrics': acadp_metrics,
        'uniform_metrics': uniform_metrics,
        'random_metrics': random_metrics,
        'improvements_vs_uniform': improvements_vs_uniform,
        'improvements_vs_random': improvements_vs_random,
    }

    return comparison


def compare_acadp_vs_baseline(
    original_df: pd.DataFrame,
    acadp_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    columns: Optional[list] = None
) -> Dict:
    """
    Compare ACADP and baseline DP results (backward compatible).

    Args:
        original_df: Original dataset
        acadp_df: ACADP-privatized dataset
        baseline_df: Baseline-privatized dataset
        columns: Columns to compare

    Returns:
        Dict with comparison metrics
    """
    logger.info("Computing ACADP utility metrics...")
    acadp_metrics = compute_all_utility_metrics(original_df, acadp_df, columns)

    logger.info("Computing baseline utility metrics...")
    baseline_metrics = compute_all_utility_metrics(original_df, baseline_df, columns)

    # Compute improvements (positive = ACADP is better)
    improvements = {}

    for metric_key in ['mean_mse', 'mean_mae', 'mean_relative_error',
                       'correlation_error', 'mean_kl_divergence',
                       'mean_js_divergence', 'mean_wasserstein_distance',
                       'mean_ks_statistic', 'mean_query_error']:
        if metric_key in acadp_metrics and metric_key in baseline_metrics:
            baseline_val = baseline_metrics[metric_key]
            acadp_val = acadp_metrics[metric_key]
            if baseline_val != 0:
                improvements[metric_key] = (baseline_val - acadp_val) / baseline_val * 100
            else:
                improvements[metric_key] = 0.0

    comparison = {
        'acadp_metrics': acadp_metrics,
        'baseline_metrics': baseline_metrics,
        'improvements': improvements,
    }

    return comparison


def run_multi_epsilon_sweep(
    original_df: pd.DataFrame,
    blocks: List[Set[str]],
    bounds: Dict[str, tuple],
    epsilon_values: List[float] = None,
    mechanism: str = 'laplace',
    allocation_method: str = 'inverse_sensitivity',
    random_state: int = 42,
    n_trials: int = 5,
    columns: Optional[list] = None
) -> Dict:
    """
    Run evaluation across multiple epsilon values.

    For each epsilon:
    - Run ACADP privatization (n_trials times, averaged)
    - Run Uniform baseline (n_trials times, averaged)
    - Run Random blocking baseline (n_trials times, averaged)
    - Compute all metrics

    Args:
        original_df: Original dataset
        blocks: Privacy blocks from correlation analysis
        bounds: Feature bounds
        epsilon_values: List of epsilon values to test
        mechanism: DP mechanism
        allocation_method: Budget allocation method
        random_state: Random seed
        n_trials: Number of trials per epsilon for averaging
        columns: Columns to evaluate

    Returns:
        Dict mapping epsilon -> comparison results
    """
    if epsilon_values is None:
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    sweep_results = {}

    for eps in epsilon_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running sweep for ε = {eps:.2f}")
        logger.info(f"{'='*60}")

        # Average over multiple trials
        acadp_dfs = []
        uniform_dfs = []
        random_dfs = []

        for trial in range(n_trials):
            trial_seed = random_state + trial * 100

            # ACADP
            privatizer = DatasetPrivatizer(
                blocks=blocks,
                bounds=bounds,
                total_epsilon=eps,
                mechanism=mechanism,
                allocation_method=allocation_method,
                random_state=trial_seed
            )
            acadp_dfs.append(privatizer.privatize(original_df))

            # Uniform baseline
            uniform_dfs.append(baseline_uniform_privatize(
                original_df, bounds, eps, mechanism=mechanism,
                random_state=trial_seed
            ))

            # Random blocking
            random_dfs.append(baseline_random_blocking_privatize(
                original_df, bounds, eps, mechanism=mechanism,
                random_state=trial_seed
            ))

        # Average the privatized DataFrames
        acadp_avg = _average_dataframes(acadp_dfs, original_df, bounds)
        uniform_avg = _average_dataframes(uniform_dfs, original_df, bounds)
        random_avg = _average_dataframes(random_dfs, original_df, bounds)

        # Compare
        comparison = compare_three_way(
            original_df, acadp_avg, uniform_avg, random_avg, columns
        )

        sweep_results[eps] = comparison
        logger.info(f"ε={eps:.2f} complete")

    return sweep_results


def _average_dataframes(
    dfs: List[pd.DataFrame],
    original_df: pd.DataFrame,
    bounds: Dict[str, tuple]
) -> pd.DataFrame:
    """Average multiple privatized DataFrames for stability."""
    avg_df = original_df.copy()
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    bounded_cols = [col for col in numeric_cols if col in bounds]

    for col in bounded_cols:
        col_values = np.stack([df[col].values for df in dfs], axis=0)
        avg_df[col] = col_values.mean(axis=0)

    return avg_df


def save_comparison(comparison: Dict, filepath: str) -> None:
    """Save comparison results to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def convert_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj) if not isinstance(obj, (int, str, bool, type(None))) else obj

    comparison_serializable = convert_types(comparison)

    with open(filepath, 'w') as f:
        json.dump(comparison_serializable, f, indent=2)

    logger.info(f"Comparison saved to {filepath}")
