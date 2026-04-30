"""
Publication-quality visualization for ACADP evaluation results.
Generates comprehensive plots for all metrics.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette for methods
METHOD_COLORS = {
    'ACADP': '#2ecc71',       # Green
    'Uniform': '#e74c3c',     # Red
    'Random': '#3498db',      # Blue
    'Original': '#95a5a6',    # Gray
}


def plot_multi_epsilon_curves(
    sweep_results: Dict,
    output_dir: str = 'output/plots'
) -> None:
    """
    Plot utility metrics across multiple epsilon values.
    One plot per metric, with lines for each method.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epsilon_values = sorted(sweep_results.keys())

    # Metrics to plot (metric_key, display_name, lower_is_better)
    metric_configs = [
        ('mean_mae', 'Mean Absolute Error', True),
        ('mean_mse', 'Mean Squared Error', True),
        ('correlation_error', 'Correlation Preservation Error', True),
        ('mean_kl_divergence', 'KL Divergence', True),
        ('mean_js_divergence', 'Jensen-Shannon Divergence', True),
        ('mean_wasserstein_distance', 'Wasserstein Distance', True),
        ('mean_ks_statistic', 'KS Statistic', True),
        ('mean_query_error', 'Mean Statistical Query Error', True),
    ]

    for metric_key, display_name, _ in metric_configs:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract values for each method
        acadp_vals = []
        uniform_vals = []
        random_vals = []

        for eps in epsilon_values:
            result = sweep_results[eps]
            acadp_vals.append(result['acadp_metrics'].get(metric_key, np.nan))
            uniform_vals.append(result['uniform_metrics'].get(metric_key, np.nan))
            random_vals.append(result['random_metrics'].get(metric_key, np.nan))

        ax.plot(epsilon_values, acadp_vals, 'o-', color=METHOD_COLORS['ACADP'],
                linewidth=2.5, markersize=8, label='ACADP', zorder=3)
        ax.plot(epsilon_values, uniform_vals, 's--', color=METHOD_COLORS['Uniform'],
                linewidth=2, markersize=7, label='Uniform Baseline', zorder=2)
        ax.plot(epsilon_values, random_vals, '^-.', color=METHOD_COLORS['Random'],
                linewidth=2, markersize=7, label='Random Blocking', zorder=2)

        ax.set_xlabel('Privacy Budget (ε)', fontsize=13)
        ax.set_ylabel(display_name, fontsize=13)
        ax.set_title(f'{display_name} vs Privacy Budget', fontsize=15, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        filename = f"epsilon_sweep_{metric_key}.png"
        plt.savefig(Path(output_dir) / filename)
        logger.info(f"Saved {filename}")
        plt.close()

    # Combined plot (2x2 grid of key metrics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    key_metrics = [
        ('mean_mae', 'Mean Absolute Error'),
        ('correlation_error', 'Correlation Error'),
        ('mean_kl_divergence', 'KL Divergence'),
        ('mean_wasserstein_distance', 'Wasserstein Distance'),
    ]

    for ax, (metric_key, display_name) in zip(axes.flatten(), key_metrics):
        acadp_vals = [sweep_results[eps]['acadp_metrics'].get(metric_key, np.nan) for eps in epsilon_values]
        uniform_vals = [sweep_results[eps]['uniform_metrics'].get(metric_key, np.nan) for eps in epsilon_values]
        random_vals = [sweep_results[eps]['random_metrics'].get(metric_key, np.nan) for eps in epsilon_values]

        ax.plot(epsilon_values, acadp_vals, 'o-', color=METHOD_COLORS['ACADP'],
                linewidth=2.5, markersize=8, label='ACADP')
        ax.plot(epsilon_values, uniform_vals, 's--', color=METHOD_COLORS['Uniform'],
                linewidth=2, markersize=7, label='Uniform')
        ax.plot(epsilon_values, random_vals, '^-.', color=METHOD_COLORS['Random'],
                linewidth=2, markersize=7, label='Random')

        ax.set_xlabel('Privacy Budget (ε)')
        ax.set_ylabel(display_name)
        ax.set_title(display_name, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.suptitle('ACADP Performance Across Privacy Budgets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'epsilon_sweep_combined.png')
    logger.info("Saved epsilon_sweep_combined.png")
    plt.close()


def plot_correlation_heatmaps(
    original_df: pd.DataFrame,
    acadp_df: pd.DataFrame,
    uniform_df: pd.DataFrame,
    columns: List[str],
    output_dir: str = 'output/plots'
) -> None:
    """Plot correlation heatmap comparison (Original / ACADP / Baseline)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    orig_corr = original_df[columns].corr()
    acadp_corr = acadp_df[columns].corr()
    uniform_corr = uniform_df[columns].corr()

    # Compute errors
    acadp_err = np.mean(np.abs(orig_corr.values - acadp_corr.values))
    uniform_err = np.mean(np.abs(orig_corr.values - uniform_corr.values))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.heatmap(orig_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Correlation'},
                square=True, linewidths=0.5)
    axes[0].set_title('Original Data', fontsize=14, fontweight='bold')

    sns.heatmap(acadp_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Correlation'},
                square=True, linewidths=0.5)
    axes[1].set_title(f'ACADP (error={acadp_err:.4f})', fontsize=14, fontweight='bold')

    sns.heatmap(uniform_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=axes[2], cbar_kws={'label': 'Correlation'},
                square=True, linewidths=0.5)
    axes[2].set_title(f'Uniform Baseline (error={uniform_err:.4f})', fontsize=14, fontweight='bold')

    plt.suptitle('Correlation Structure Preservation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'correlation_heatmaps.png')
    logger.info("Saved correlation_heatmaps.png")
    plt.close()


def plot_per_feature_comparison(
    comparison: Dict,
    output_dir: str = 'output/plots'
) -> None:
    """Plot per-feature bar charts comparing ACADP vs baselines."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for metric_key, display_name in [('mae', 'Mean Absolute Error'),
                                     ('mse', 'Mean Squared Error'),
                                     ('wasserstein_distance', 'Wasserstein Distance'),
                                     ('kl_divergence', 'KL Divergence')]:

        acadp_vals = comparison['acadp_metrics'].get(metric_key, {})
        uniform_vals = comparison['uniform_metrics'].get(metric_key, {})
        random_vals = comparison.get('random_metrics', {}).get(metric_key, {})

        if not acadp_vals:
            continue

        features = sorted(set(acadp_vals.keys()) & set(uniform_vals.keys()))
        if not features:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(features) * 1.5), 7))
        x = np.arange(len(features))
        width = 0.25

        bars1 = ax.bar(x - width, [acadp_vals.get(f, 0) for f in features],
                       width, label='ACADP', color=METHOD_COLORS['ACADP'], alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x, [uniform_vals.get(f, 0) for f in features],
                       width, label='Uniform', color=METHOD_COLORS['Uniform'], alpha=0.85, edgecolor='white')
        if random_vals:
            bars3 = ax.bar(x + width, [random_vals.get(f, 0) for f in features],
                           width, label='Random', color=METHOD_COLORS['Random'], alpha=0.85, edgecolor='white')

        ax.set_xlabel('Feature', fontsize=13)
        ax.set_ylabel(display_name, fontsize=13)
        ax.set_title(f'{display_name} per Feature', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'per_feature_{metric_key}.png')
        logger.info(f"Saved per_feature_{metric_key}.png")
        plt.close()


def plot_improvement_summary(
    comparison: Dict,
    output_dir: str = 'output/plots'
) -> None:
    """Plot improvement percentage bar chart."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    improvements_uniform = comparison.get('improvements_vs_uniform', comparison.get('improvements', {}))
    improvements_random = comparison.get('improvements_vs_random', {})

    if not improvements_uniform:
        return

    metric_labels = {
        'mean_mse': 'MSE',
        'mean_mae': 'MAE',
        'mean_relative_error': 'Relative Error',
        'correlation_error': 'Correlation Error',
        'mean_kl_divergence': 'KL Divergence',
        'mean_js_divergence': 'JS Divergence',
        'mean_wasserstein_distance': 'Wasserstein Dist.',
        'mean_ks_statistic': 'KS Statistic',
        'mean_query_error': 'Query Error',
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = [k for k in metric_labels if k in improvements_uniform]
    labels = [metric_labels[k] for k in metrics]
    values_uniform = [improvements_uniform[k] for k in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, values_uniform, width, label='vs Uniform Baseline',
                   color=METHOD_COLORS['Uniform'], alpha=0.85, edgecolor='white')

    if improvements_random:
        values_random = [improvements_random.get(k, 0) for k in metrics]
        bars2 = ax.bar(x + width/2, values_random, width, label='vs Random Blocking',
                       color=METHOD_COLORS['Random'], alpha=0.85, edgecolor='white')

    ax.set_xlabel('Metric', fontsize=13)
    ax.set_ylabel('ACADP Improvement (%)', fontsize=13)
    ax.set_title('ACADP Improvement Over Baselines', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'improvement_summary.png')
    logger.info("Saved improvement_summary.png")
    plt.close()


def plot_distribution_comparison(
    original_df: pd.DataFrame,
    acadp_df: pd.DataFrame,
    uniform_df: pd.DataFrame,
    columns: List[str],
    output_dir: str = 'output/plots',
    max_features: int = 8
) -> None:
    """Plot distribution comparisons (overlaid histograms) for key features."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_features = min(len(columns), max_features)
    cols_to_plot = columns[:n_features]

    n_cols = 2
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(cols_to_plot):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx]

        ax.hist(original_df[col].dropna(), bins=50, alpha=0.5, density=True,
                color=METHOD_COLORS['Original'], label='Original', edgecolor='none')
        ax.hist(acadp_df[col].dropna(), bins=50, alpha=0.5, density=True,
                color=METHOD_COLORS['ACADP'], label='ACADP', edgecolor='none')
        ax.hist(uniform_df[col].dropna(), bins=50, alpha=0.4, density=True,
                color=METHOD_COLORS['Uniform'], label='Uniform', edgecolor='none')

        ax.set_title(col, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylabel('Density')

    # Hide empty subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].set_visible(False)

    plt.suptitle('Distribution Comparison: Original vs Privatized', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'distribution_comparison.png')
    logger.info("Saved distribution_comparison.png")
    plt.close()


def plot_budget_allocation(
    blocks: List[Set[str]],
    allocations: Dict[int, float],
    total_epsilon: float,
    output_dir: str = 'output/plots'
) -> None:
    """Plot privacy budget allocation across blocks."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Pie chart
    block_labels = [f"Block {i}\n({len(blocks[i])} features)" for i in range(len(blocks))]
    eps_values = [allocations.get(i, 0) for i in range(len(blocks))]
    colors = plt.cm.Set3(np.linspace(0, 1, len(blocks)))

    wedges, texts, autotexts = ax1.pie(
        eps_values, labels=block_labels, autopct='%1.1f%%',
        colors=colors, pctdistance=0.85, startangle=90
    )
    ax1.set_title(f'Privacy Budget Allocation (ε = {total_epsilon})', fontweight='bold')

    # Bar chart with block details
    block_indices = list(range(len(blocks)))
    block_sizes = [len(blocks[i]) for i in block_indices]
    eps_vals = [allocations.get(i, 0) for i in block_indices]

    bars = ax2.bar(block_indices, eps_vals, color=colors, edgecolor='white', alpha=0.85)
    ax2.set_xlabel('Block Index')
    ax2.set_ylabel('Allocated ε')
    ax2.set_title('ε Allocation per Block', fontweight='bold')

    # Add feature count labels
    for bar, size in zip(bars, block_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{size} feat.', ha='center', va='bottom', fontsize=9)

    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'budget_allocation.png')
    logger.info("Saved budget_allocation.png")
    plt.close()


def plot_downstream_ml_results(
    ml_results: Dict,
    output_dir: str = 'output/plots'
) -> None:
    """Plot downstream ML performance comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Classification results
    clf_data = {}
    for method, results in ml_results.items():
        if 'classification' in results and results['classification']:
            clf_data[method] = results['classification']

    if clf_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        methods = list(clf_data.keys())
        accuracies = [clf_data[m].get('privatized_accuracy', 0) for m in methods]
        f1_scores = [clf_data[m].get('privatized_f1', 0) for m in methods]

        # Get original accuracy (same for all)
        orig_acc = clf_data[methods[0]].get('original_accuracy', 0) if methods else 0
        orig_f1 = clf_data[methods[0]].get('original_f1', 0) if methods else 0

        colors = [METHOD_COLORS.get(m.upper(), '#95a5a6') for m in methods]

        # Accuracy
        x = np.arange(len(methods))
        bars = ax1.bar(x, accuracies, color=[METHOD_COLORS.get('ACADP' if 'acadp' in m.lower() else
                       'Uniform' if 'uniform' in m.lower() else 'Random', '#95a5a6') for m in methods],
                       alpha=0.85, edgecolor='white')
        ax1.axhline(y=orig_acc, color='gray', linestyle='--', linewidth=2, label=f'Original ({orig_acc:.3f})')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # F1
        bars = ax2.bar(x, f1_scores, color=[METHOD_COLORS.get('ACADP' if 'acadp' in m.lower() else
                       'Uniform' if 'uniform' in m.lower() else 'Random', '#95a5a6') for m in methods],
                       alpha=0.85, edgecolor='white')
        ax2.axhline(y=orig_f1, color='gray', linestyle='--', linewidth=2, label=f'Original ({orig_f1:.3f})')
        ax2.set_ylabel('F1 Score (Weighted)')
        ax2.set_title('Classification F1 Score', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Downstream ML: Classification Performance', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'downstream_classification.png')
        logger.info("Saved downstream_classification.png")
        plt.close()

    # Regression results
    reg_data = {}
    for method, results in ml_results.items():
        if 'regression' in results and results['regression']:
            reg_data[method] = results['regression']

    if reg_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        methods = list(reg_data.keys())
        rmse_vals = [reg_data[m].get('privatized_rmse', 0) for m in methods]
        r2_vals = [reg_data[m].get('privatized_r2', 0) for m in methods]

        orig_rmse = reg_data[methods[0]].get('original_rmse', 0) if methods else 0
        orig_r2 = reg_data[methods[0]].get('original_r2', 0) if methods else 0

        x = np.arange(len(methods))

        # RMSE (lower is better)
        bars = ax1.bar(x, rmse_vals, color=[METHOD_COLORS.get('ACADP' if 'acadp' in m.lower() else
                       'Uniform' if 'uniform' in m.lower() else 'Random', '#95a5a6') for m in methods],
                       alpha=0.85, edgecolor='white')
        ax1.axhline(y=orig_rmse, color='gray', linestyle='--', linewidth=2, label=f'Original ({orig_rmse:.2f})')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Regression RMSE (lower is better)', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # R²  (higher is better)
        bars = ax2.bar(x, r2_vals, color=[METHOD_COLORS.get('ACADP' if 'acadp' in m.lower() else
                       'Uniform' if 'uniform' in m.lower() else 'Random', '#95a5a6') for m in methods],
                       alpha=0.85, edgecolor='white')
        ax2.axhline(y=orig_r2, color='gray', linestyle='--', linewidth=2, label=f'Original ({orig_r2:.3f})')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Regression R² (higher is better)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle('Downstream ML: Regression Performance', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'downstream_regression.png')
        logger.info("Saved downstream_regression.png")
        plt.close()


def generate_all_plots(
    original_df: pd.DataFrame,
    acadp_df: pd.DataFrame,
    uniform_df: pd.DataFrame,
    random_df: pd.DataFrame,
    comparison: Dict,
    blocks: List[Set[str]],
    allocations: Dict[int, float],
    total_epsilon: float,
    columns: List[str],
    sweep_results: Optional[Dict] = None,
    ml_results: Optional[Dict] = None,
    output_dir: str = 'output/plots'
) -> None:
    """Generate ALL plots in one call."""
    logger.info(f"Generating all plots to {output_dir}")

    plot_correlation_heatmaps(original_df, acadp_df, uniform_df, columns, output_dir)
    plot_per_feature_comparison(comparison, output_dir)
    plot_improvement_summary(comparison, output_dir)
    plot_distribution_comparison(original_df, acadp_df, uniform_df, columns, output_dir)
    plot_budget_allocation(blocks, allocations, total_epsilon, output_dir)

    if sweep_results:
        plot_multi_epsilon_curves(sweep_results, output_dir)

    if ml_results:
        plot_downstream_ml_results(ml_results, output_dir)

    logger.info(f"All plots saved to {output_dir}")
