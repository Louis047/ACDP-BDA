"""
Approximate Pearson correlation computation.
Efficient correlation estimation for large-scale datasets.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        Pearson correlation coefficient (-1 to 1)
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.nan
    
    # Compute correlation
    corr = np.corrcoef(x_clean, y_clean)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def approximate_correlation_matrix(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute correlation matrix, optionally on a sample.
    
    Args:
        df: DataFrame to analyze
        sample_size: If provided, sample this many rows before computing
        columns: Columns to include (None = all numeric)
    
    Returns:
        Correlation matrix DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) < 2:
        logger.warning("Need at least 2 numeric columns for correlation")
        return pd.DataFrame()
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} rows for correlation computation")
        df_sample = df[columns].sample(n=min(sample_size, len(df)), random_state=42)
    else:
        df_sample = df[columns]
    
    logger.info(f"Computing correlation matrix for {len(columns)} columns")
    corr_matrix = df_sample.corr()
    
    return corr_matrix


def pairwise_correlations(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    threshold: float = 0.0
) -> list:
    """
    Get pairwise correlations above threshold.
    
    Args:
        df: DataFrame to analyze
        columns: Columns to consider (None = all numeric)
        threshold: Minimum absolute correlation to include
    
    Returns:
        List of tuples: (col1, col2, correlation)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = approximate_correlation_matrix(df, columns=columns)
    
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                pairs.append((col_i, col_j, float(corr_val)))
    
    logger.info(f"Found {len(pairs)} pairs with |correlation| >= {threshold}")
    return pairs
