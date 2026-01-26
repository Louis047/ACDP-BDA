"""
Compare ACADP vs baseline DP.
Provides comprehensive comparison metrics.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional
import logging

from .utility_metrics import compute_all_utility_metrics
from .baselines import baseline_privatize

logger = logging.getLogger(__name__)


def compare_acadp_vs_baseline(
    original_df: pd.DataFrame,
    acadp_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    columns: Optional[list] = None
) -> Dict:
    """
    Compare ACADP and baseline DP results.
    
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
    
    # Compute improvements
    improvements = {}
    
    if 'mean_mse' in acadp_metrics and 'mean_mse' in baseline_metrics:
        mse_improvement = (baseline_metrics['mean_mse'] - acadp_metrics['mean_mse']) / baseline_metrics['mean_mse'] * 100
        improvements['mse_improvement_pct'] = mse_improvement
    
    if 'mean_mae' in acadp_metrics and 'mean_mae' in baseline_metrics:
        mae_improvement = (baseline_metrics['mean_mae'] - acadp_metrics['mean_mae']) / baseline_metrics['mean_mae'] * 100
        improvements['mae_improvement_pct'] = mae_improvement
    
    if 'mean_correlation_diff' in acadp_metrics and 'mean_correlation_diff' in baseline_metrics:
        corr_improvement = (baseline_metrics['mean_correlation_diff'] - acadp_metrics['mean_correlation_diff']) / baseline_metrics['mean_correlation_diff'] * 100
        improvements['correlation_preservation_improvement_pct'] = corr_improvement
    
    comparison = {
        'acadp_metrics': acadp_metrics,
        'baseline_metrics': baseline_metrics,
        'improvements': improvements
    }
    
    return comparison


def save_comparison(comparison: Dict, filepath: str) -> None:
    """Save comparison results to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return float(obj) if isinstance(obj, float) else int(obj)
        else:
            return str(obj)
    
    comparison_serializable = convert_types(comparison)
    
    with open(filepath, 'w') as f:
        json.dump(comparison_serializable, f, indent=2)
    
    logger.info(f"Comparison saved to {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare ACADP vs baseline")
    parser.add_argument("--original", required=True, help="Original dataset Parquet")
    parser.add_argument("--acadp", required=True, help="ACADP-privatized dataset Parquet")
    parser.add_argument("--baseline", required=True, help="Baseline-privatized dataset Parquet")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load datasets
    from ..ingestion.load_data import load_parquet
    
    original_df = load_parquet(args.original)
    acadp_df = load_parquet(args.acadp)
    baseline_df = load_parquet(args.baseline)
    
    # Compare
    comparison = compare_acadp_vs_baseline(original_df, acadp_df, baseline_df)
    
    # Save
    save_comparison(comparison, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    if 'improvements' in comparison:
        for metric, value in comparison['improvements'].items():
            print(f"{metric}: {value:.2f}%")
    print("=" * 60)
