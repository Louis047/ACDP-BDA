"""
Utility metrics for evaluating privatized datasets.
Measures error, variance, and correlation preservation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def mean_squared_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute Mean Squared Error (MSE) per feature.
    
    Args:
        original: Original DataFrame
        privatized: Privatized DataFrame
        columns: Columns to evaluate (None = all common columns)
    
    Returns:
        Dict mapping column names to MSE
    """
    if columns is None:
        columns = list(set(original.columns) & set(privatized.columns))
    
    mse_dict = {}
    
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        
        orig_values = original[col].values.astype(float)
        priv_values = privatized[col].values.astype(float)
        
        # Handle different lengths
        min_len = min(len(orig_values), len(priv_values))
        orig_values = orig_values[:min_len]
        priv_values = priv_values[:min_len]
        
        mse = np.mean((orig_values - priv_values) ** 2)
        mse_dict[col] = float(mse)
    
    return mse_dict


def mean_absolute_error(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute Mean Absolute Error (MAE) per feature.
    
    Args:
        original: Original DataFrame
        privatized: Privatized DataFrame
        columns: Columns to evaluate
    
    Returns:
        Dict mapping column names to MAE
    """
    if columns is None:
        columns = list(set(original.columns) & set(privatized.columns))
    
    mae_dict = {}
    
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        
        orig_values = original[col].values.astype(float)
        priv_values = privatized[col].values.astype(float)
        
        min_len = min(len(orig_values), len(priv_values))
        orig_values = orig_values[:min_len]
        priv_values = priv_values[:min_len]
        
        mae = np.mean(np.abs(orig_values - priv_values))
        mae_dict[col] = float(mae)
    
    return mae_dict


def variance_preservation(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Measure variance preservation ratio.
    
    Returns ratio of privatized variance to original variance.
    Values close to 1.0 indicate good preservation.
    
    Args:
        original: Original DataFrame
        privatized: Privatized DataFrame
        columns: Columns to evaluate
    
    Returns:
        Dict mapping column names to variance ratio
    """
    if columns is None:
        columns = list(set(original.columns) & set(privatized.columns))
    
    variance_ratios = {}
    
    for col in columns:
        if col not in original.columns or col not in privatized.columns:
            continue
        
        orig_var = original[col].var()
        priv_var = privatized[col].var()
        
        if orig_var == 0:
            variance_ratios[col] = 1.0 if priv_var == 0 else np.inf
        else:
            variance_ratios[col] = float(priv_var / orig_var)
    
    return variance_ratios


def correlation_preservation(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[Tuple[str, str], float]:
    """
    Measure correlation preservation.
    
    Computes correlation in both datasets and compares.
    
    Args:
        original: Original DataFrame
        privatized: Privatized DataFrame
        columns: Columns to evaluate
    
    Returns:
        Dict mapping (col1, col2) tuples to correlation difference
    """
    if columns is None:
        columns = list(set(original.columns) & set(privatized.columns))
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(original[col])]
    
    if len(columns) < 2:
        return {}
    
    # Compute correlation matrices
    orig_corr = original[columns].corr()
    priv_corr = privatized[columns].corr()
    
    # Compute differences
    corr_diff = {}
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            orig_corr_val = orig_corr.loc[col1, col2]
            priv_corr_val = priv_corr.loc[col1, col2]
            
            # Absolute difference
            diff = abs(orig_corr_val - priv_corr_val)
            corr_diff[(col1, col2)] = float(diff)
    
    return corr_diff


def compute_all_utility_metrics(
    original: pd.DataFrame,
    privatized: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict:
    """
    Compute all utility metrics.
    
    Args:
        original: Original DataFrame
        privatized: Privatized DataFrame
        columns: Columns to evaluate
    
    Returns:
        Dict containing all metrics
    """
    metrics = {
        'mse': mean_squared_error(original, privatized, columns),
        'mae': mean_absolute_error(original, privatized, columns),
        'variance_ratios': variance_preservation(original, privatized, columns),
        'correlation_differences': correlation_preservation(original, privatized, columns)
    }
    
    # Aggregate statistics
    if metrics['mse']:
        metrics['mean_mse'] = np.mean(list(metrics['mse'].values()))
        metrics['median_mse'] = np.median(list(metrics['mse'].values()))
    
    if metrics['mae']:
        metrics['mean_mae'] = np.mean(list(metrics['mae'].values()))
        metrics['median_mae'] = np.median(list(metrics['mae'].values()))
    
    if metrics['variance_ratios']:
        metrics['mean_variance_ratio'] = np.mean(list(metrics['variance_ratios'].values()))
    
    if metrics['correlation_differences']:
        metrics['mean_correlation_diff'] = np.mean(list(metrics['correlation_differences'].values()))
        metrics['median_correlation_diff'] = np.median(list(metrics['correlation_differences'].values()))
    
    return metrics
