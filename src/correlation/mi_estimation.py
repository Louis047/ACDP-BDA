"""
Discretized Mutual Information estimation.
Measures non-linear dependencies between features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import logging

logger = logging.getLogger(__name__)


def discretize_continuous(
    x: np.ndarray,
    n_bins: int = 10
) -> np.ndarray:
    """
    Discretize continuous variable into bins.
    
    Args:
        x: Continuous variable
        n_bins: Number of bins
    
    Returns:
        Discretized variable
    """
    # Remove NaN
    mask = ~np.isnan(x)
    x_clean = x[mask]
    
    if len(x_clean) == 0:
        return np.full_like(x, np.nan, dtype=int)
    
    # Create bins
    _, bin_edges = np.histogram(x_clean, bins=n_bins)
    discretized = np.digitize(x, bin_edges)
    
    return discretized


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    discrete_x: bool = False,
    discrete_y: bool = False,
    n_bins: int = 10
) -> float:
    """
    Estimate mutual information between two variables.
    
    Args:
        x: First variable
        y: Second variable
        discrete_x: Whether x is already discrete
        discrete_y: Whether y is already discrete
        n_bins: Number of bins for discretization
    
    Returns:
        Mutual information (non-negative)
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return 0.0
    
    # Discretize if needed
    if not discrete_x:
        x_clean = discretize_continuous(x_clean, n_bins=n_bins)
    if not discrete_y:
        y_clean = discretize_continuous(y_clean, n_bins=n_bins)
    
    # Remove any remaining NaN from discretization
    mask = ~(np.isnan(x_clean) | np.isnan(y_clean))
    x_clean = x_clean[mask]
    y_clean = y_clean[mask]
    
    if len(x_clean) < 2:
        return 0.0
    
    # Use sklearn's MI estimation
    try:
        # Reshape for sklearn
        x_reshaped = x_clean.reshape(-1, 1)
        
        # Determine if target is discrete
        y_is_discrete = discrete_y or (y_clean.dtype in [int, np.int64] and y_clean.nunique() < 20)
        
        if y_is_discrete:
            mi = mutual_info_classif(x_reshaped, y_clean, random_state=42, discrete_features=[True])[0]
        else:
            mi = mutual_info_regression(x_reshaped, y_clean, random_state=42, discrete_features=[True])[0]
        
        return float(mi) if not np.isnan(mi) else 0.0
    except Exception as e:
        logger.warning(f"MI estimation failed: {e}. Returning 0.0")
        return 0.0


def pairwise_mutual_information(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    threshold: float = 0.0,
    n_bins: int = 10
) -> list:
    """
    Compute pairwise mutual information.
    
    Args:
        df: DataFrame to analyze
        columns: Columns to consider (None = all numeric)
        threshold: Minimum MI to include
        n_bins: Number of bins for discretization
    
    Returns:
        List of tuples: (col1, col2, MI)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) < 2:
        return []
    
    logger.info(f"Computing pairwise MI for {len(columns)} columns")
    
    pairs = []
    n_pairs = len(columns) * (len(columns) - 1) // 2
    
    for i, col_i in enumerate(columns):
        for col_j in columns[i + 1:]:
            mi_val = mutual_information(
                df[col_i].values,
                df[col_j].values,
                n_bins=n_bins
            )
            
            if mi_val >= threshold:
                pairs.append((col_i, col_j, mi_val))
    
    logger.info(f"Found {len(pairs)} pairs with MI >= {threshold}")
    return pairs
