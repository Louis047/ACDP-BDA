"""
Utility metrics for evaluating privatized datasets.
Measures error, variance, correlation preservation, distributional distance,
and statistical query accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


def mean_squared_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute Mean Squared Error (MSE) per feature."""
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    mse_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        orig_values = original[col].values.astype(float)
        priv_values = privatized[col].values.astype(float)
        min_len = min(len(orig_values), len(priv_values))
        mse = np.mean((orig_values[:min_len] - priv_values[:min_len]) ** 2)
        mse_dict[col] = float(mse)

    return mse_dict


def mean_absolute_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute Mean Absolute Error (MAE) per feature."""
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    mae_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        orig_values = original[col].values.astype(float)
        priv_values = privatized[col].values.astype(float)
        min_len = min(len(orig_values), len(priv_values))
        mae = np.mean(np.abs(orig_values[:min_len] - priv_values[:min_len]))
        mae_dict[col] = float(mae)

    return mae_dict


def relative_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute Mean Relative Error per feature (normalized by original range)."""
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    re_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        orig_values = original[col].values.astype(float)
        priv_values = privatized[col].values.astype(float)
        min_len = min(len(orig_values), len(priv_values))

        orig_range = orig_values[:min_len].max() - orig_values[:min_len].min()
        if orig_range == 0:
            re_dict[col] = 0.0
        else:
            mae = np.mean(np.abs(orig_values[:min_len] - priv_values[:min_len]))
            re_dict[col] = float(mae / orig_range)

    return re_dict


def variance_preservation(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Measure variance preservation ratio.
    Values close to 1.0 indicate good preservation.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    variance_ratios = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        orig_var = original[col].var()
        priv_var = privatized[col].var()
        if orig_var == 0:
            variance_ratios[col] = 1.0 if priv_var == 0 else float('inf')
        else:
            variance_ratios[col] = float(priv_var / orig_var)

    return variance_ratios


def correlation_preservation_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> float:
    """
    Compute average absolute correlation matrix error (Frobenius-style).
    Lower is better.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    if len(columns) < 2:
        return 0.0

    orig_corr = original[columns].corr().values
    priv_corr = privatized[columns].corr().values

    # Mean absolute element-wise difference
    return float(np.mean(np.abs(orig_corr - priv_corr)))


def correlation_preservation_per_pair(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[Tuple[str, str], float]:
    """Compute correlation difference per feature pair."""
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    if len(columns) < 2:
        return {}

    orig_corr = original[columns].corr()
    priv_corr = privatized[columns].corr()

    corr_diff = {}
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            diff = abs(orig_corr.loc[col1, col2] - priv_corr.loc[col1, col2])
            corr_diff[(col1, col2)] = float(diff)

    return corr_diff


def kl_divergence_per_feature(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_bins: int = 50
) -> Dict[str, float]:
    """
    Compute KL Divergence per feature using binned histograms.
    KL(original || privatized). Lower is better.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    kl_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue

        orig_vals = original[col].dropna().values
        priv_vals = privatized[col].dropna().values

        if len(orig_vals) == 0 or len(priv_vals) == 0:
            kl_dict[col] = float('inf')
            continue

        # Use common bin edges from original data range
        min_val = min(orig_vals.min(), priv_vals.min())
        max_val = max(orig_vals.max(), priv_vals.max())
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        orig_hist, _ = np.histogram(orig_vals, bins=bin_edges, density=True)
        priv_hist, _ = np.histogram(priv_vals, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0) and division by zero
        eps = 1e-10
        orig_hist = orig_hist + eps
        priv_hist = priv_hist + eps

        # Normalize to proper probability distributions
        orig_hist = orig_hist / orig_hist.sum()
        priv_hist = priv_hist / priv_hist.sum()

        kl = float(scipy_stats.entropy(orig_hist, priv_hist))
        kl_dict[col] = kl

    return kl_dict


def js_divergence_per_feature(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_bins: int = 50
) -> Dict[str, float]:
    """
    Compute Jensen-Shannon Divergence per feature.
    Symmetric, bounded [0, ln(2)]. Lower is better.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    js_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue

        orig_vals = original[col].dropna().values
        priv_vals = privatized[col].dropna().values

        if len(orig_vals) == 0 or len(priv_vals) == 0:
            js_dict[col] = 1.0
            continue

        min_val = min(orig_vals.min(), priv_vals.min())
        max_val = max(orig_vals.max(), priv_vals.max())
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        orig_hist, _ = np.histogram(orig_vals, bins=bin_edges)
        priv_hist, _ = np.histogram(priv_vals, bins=bin_edges)

        # Normalize
        orig_hist = orig_hist.astype(float) + 1e-10
        priv_hist = priv_hist.astype(float) + 1e-10
        orig_hist /= orig_hist.sum()
        priv_hist /= priv_hist.sum()

        js = float(jensenshannon(orig_hist, priv_hist))
        js_dict[col] = js

    return js_dict


def wasserstein_distance_per_feature(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute Wasserstein (Earth Mover's) distance per feature.
    Lower is better.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    wd_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue

        orig_vals = original[col].dropna().values
        priv_vals = privatized[col].dropna().values

        if len(orig_vals) == 0 or len(priv_vals) == 0:
            wd_dict[col] = float('inf')
            continue

        wd = float(scipy_stats.wasserstein_distance(orig_vals, priv_vals))
        wd_dict[col] = wd

    return wd_dict


def ks_statistic_per_feature(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute Kolmogorov-Smirnov statistic per feature.
    Max absolute CDF difference. Lower is better.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    ks_dict = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue

        orig_vals = original[col].dropna().values
        priv_vals = privatized[col].dropna().values

        if len(orig_vals) == 0 or len(priv_vals) == 0:
            ks_dict[col] = 1.0
            continue

        ks_stat, _ = scipy_stats.ks_2samp(orig_vals, priv_vals)
        ks_dict[col] = float(ks_stat)

    return ks_dict


def statistical_query_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None,
    group_col: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute error on statistical queries (MEAN, SUM, STD).
    Reports relative error for each statistic per feature.
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    results = {}
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue

        orig_vals = original[col].dropna()
        priv_vals = privatized[col].dropna()

        if len(orig_vals) == 0:
            continue

        # Compute statistics on both
        orig_mean = orig_vals.mean()
        priv_mean = priv_vals.mean()
        orig_sum = orig_vals.sum()
        priv_sum = priv_vals.sum()
        orig_std = orig_vals.std()
        priv_std = priv_vals.std()
        orig_median = orig_vals.median()
        priv_median = priv_vals.median()

        def _relative_error(true_val, est_val):
            if abs(true_val) < 1e-10:
                return 0.0 if abs(est_val) < 1e-10 else float('inf')
            return abs(true_val - est_val) / abs(true_val)

        results[col] = {
            'mean_relative_error': _relative_error(orig_mean, priv_mean),
            'sum_relative_error': _relative_error(orig_sum, priv_sum),
            'std_relative_error': _relative_error(orig_std, priv_std),
            'median_relative_error': _relative_error(orig_median, priv_median),
        }

    return results


def compute_all_utility_metrics(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict:
    """
    Compute ALL utility metrics comprehensively.

    Args:
        original: Original DataFrame
        privatized: Privatized DataFrame
        columns: Columns to evaluate

    Returns:
        Dict containing all metrics with per-feature and aggregate values
    """
    if columns is None:
        columns = _get_common_numeric_columns(original, privatized)

    metrics = {}

    # Error metrics
    metrics['mse'] = mean_squared_error(original, privatized, columns)
    metrics['mae'] = mean_absolute_error(original, privatized, columns)
    metrics['relative_error'] = relative_error(original, privatized, columns)

    # Variance preservation
    metrics['variance_ratios'] = variance_preservation(original, privatized, columns)

    # Correlation preservation
    metrics['correlation_error'] = correlation_preservation_error(original, privatized, columns)
    metrics['correlation_differences'] = correlation_preservation_per_pair(original, privatized, columns)

    # Distributional metrics
    metrics['kl_divergence'] = kl_divergence_per_feature(original, privatized, columns)
    metrics['js_divergence'] = js_divergence_per_feature(original, privatized, columns)
    metrics['wasserstein_distance'] = wasserstein_distance_per_feature(original, privatized, columns)
    metrics['ks_statistic'] = ks_statistic_per_feature(original, privatized, columns)

    # Statistical query error
    metrics['statistical_query_error'] = statistical_query_error(original, privatized, columns)

    # Aggregate statistics
    for metric_name in ['mse', 'mae', 'relative_error', 'kl_divergence',
                        'js_divergence', 'wasserstein_distance', 'ks_statistic']:
        vals = metrics[metric_name]
        if vals:
            values = list(vals.values())
            metrics[f'mean_{metric_name}'] = float(np.mean(values))
            metrics[f'median_{metric_name}'] = float(np.median(values))

    # Variance ratio aggregate (how close to 1.0 on average)
    if metrics['variance_ratios']:
        vr_vals = list(metrics['variance_ratios'].values())
        # Distance from 1.0 (perfect preservation)
        metrics['mean_variance_deviation'] = float(np.mean([abs(v - 1.0) for v in vr_vals]))

    # Statistical query aggregate
    if metrics['statistical_query_error']:
        mean_errors = [v['mean_relative_error'] for v in metrics['statistical_query_error'].values()]
        metrics['mean_query_error'] = float(np.mean(mean_errors))

    return metrics


def _get_common_numeric_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> List[str]:
    """Get common numeric columns between two DataFrames."""
    common = list(set(df1.columns) & set(df2.columns))
    return [col for col in common if pd.api.types.is_numeric_dtype(df1[col])]
