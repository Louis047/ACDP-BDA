"""
Result visualization for evaluation metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_mse_comparison(
    acadp_mse: Dict[str, float],
    baseline_mse: Dict[str, float],
    output_path: Optional[str] = None
) -> None:
    """
    Plot MSE comparison between ACADP and baseline.
    
    Args:
        acadp_mse: Dict mapping column names to MSE (ACADP)
        baseline_mse: Dict mapping column names to MSE (baseline)
        output_path: Optional path to save figure
    """
    # Get common columns
    common_cols = set(acadp_mse.keys()) & set(baseline_mse.keys())
    
    if not common_cols:
        logger.warning("No common columns for MSE comparison")
        return
    
    # Prepare data
    data = {
        'Feature': list(common_cols),
        'ACADP': [acadp_mse[col] for col in common_cols],
        'Baseline': [baseline_mse[col] for col in common_cols]
    }
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['ACADP'], width, label='ACADP', alpha=0.8)
    ax.bar(x + width/2, df['Baseline'], width, label='Baseline', alpha=0.8)
    
    ax.set_xlabel('Feature')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Comparison: ACADP vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Feature'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"MSE comparison plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_heatmap(
    original_corr: pd.DataFrame,
    privatized_corr: pd.DataFrame,
    output_path: Optional[str] = None
) -> None:
    """
    Plot correlation heatmaps for original and privatized data.
    
    Args:
        original_corr: Correlation matrix (original)
        privatized_corr: Correlation matrix (privatized)
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original correlation
    sns.heatmap(original_corr, annot=False, cmap='coolwarm', center=0,
                square=True, ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Original Data Correlation')
    
    # Privatized correlation
    sns.heatmap(privatized_corr, annot=False, cmap='coolwarm', center=0,
                square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Privatized Data Correlation')
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation heatmap saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_utility_summary(
    comparison: Dict,
    output_path: Optional[str] = None
) -> None:
    """
    Plot summary of utility metrics.
    
    Args:
        comparison: Comparison dict from compare_acadp_vs_baseline
        output_path: Optional path to save figure
    """
    acadp = comparison.get('acadp_metrics', {})
    baseline = comparison.get('baseline_metrics', {})
    
    metrics_to_plot = []
    acadp_values = []
    baseline_values = []
    
    if 'mean_mse' in acadp and 'mean_mse' in baseline:
        metrics_to_plot.append('Mean MSE')
        acadp_values.append(acadp['mean_mse'])
        baseline_values.append(baseline['mean_mse'])
    
    if 'mean_mae' in acadp and 'mean_mae' in baseline:
        metrics_to_plot.append('Mean MAE')
        acadp_values.append(acadp['mean_mae'])
        baseline_values.append(baseline['mean_mae'])
    
    if 'mean_correlation_diff' in acadp and 'mean_correlation_diff' in baseline:
        metrics_to_plot.append('Mean Corr. Diff')
        acadp_values.append(acadp['mean_correlation_diff'])
        baseline_values.append(baseline['mean_correlation_diff'])
    
    if not metrics_to_plot:
        logger.warning("No metrics to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    ax.bar(x - width/2, acadp_values, width, label='ACADP', alpha=0.8)
    ax.bar(x + width/2, baseline_values, width, label='Baseline', alpha=0.8)
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Utility Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Utility summary plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
