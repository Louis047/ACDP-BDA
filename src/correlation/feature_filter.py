"""
Dimensionality reduction and feature pruning.
Removes low-variance and redundant features before correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)


def filter_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
    columns: Optional[List[str]] = None
) -> List[str]:
    """
    Identify columns with low variance (likely uninformative).
    
    Args:
        df: DataFrame to analyze
        threshold: Variance threshold (relative to max variance)
        columns: Columns to check (None = all numeric)
    
    Returns:
        List of column names to remove
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        return []
    
    variances = df[columns].var()
    max_variance = variances.max()
    
    if max_variance == 0:
        logger.warning("All columns have zero variance")
        return []
    
    # Normalize variances
    normalized_variances = variances / max_variance
    
    low_variance_cols = normalized_variances[normalized_variances < threshold].index.tolist()
    
    if low_variance_cols:
        logger.info(f"Identified {len(low_variance_cols)} low-variance columns to remove")
    
    return low_variance_cols


def filter_constant_features(df: pd.DataFrame) -> List[str]:
    """
    Identify constant features (zero variance).
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        List of constant column names
    """
    constant_cols = []
    
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        logger.info(f"Identified {len(constant_cols)} constant columns")
    
    return constant_cols


def filter_highly_correlated(
    df: pd.DataFrame,
    threshold: float = 0.95,
    columns: Optional[List[str]] = None
) -> List[str]:
    """
    Identify highly correlated features (redundant).
    Keeps one feature from each highly correlated pair.
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold (absolute value)
        columns: Columns to check (None = all numeric)
    
    Returns:
        List of column names to remove (one from each highly correlated pair)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) < 2:
        return []
    
    # Compute correlation matrix
    corr_matrix = df[columns].corr().abs()
    
    # Find highly correlated pairs
    to_remove = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                
                # Remove the one with lower variance (less informative)
                var_i = df[col_i].var()
                var_j = df[col_j].var()
                
                if var_i <= var_j:
                    to_remove.add(col_i)
                else:
                    to_remove.add(col_j)
    
    to_remove_list = list(to_remove)
    
    if to_remove_list:
        logger.info(f"Identified {len(to_remove_list)} highly correlated columns to remove")
    
    return to_remove_list


def apply_feature_filters(
    df: pd.DataFrame,
    remove_low_variance: bool = True,
    remove_constant: bool = True,
    remove_highly_correlated: bool = False,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Apply all feature filters.
    
    Args:
        df: DataFrame to filter
        remove_low_variance: Remove low-variance features
        remove_constant: Remove constant features
        remove_highly_correlated: Remove redundant highly correlated features
        variance_threshold: Variance threshold for low-variance filter
        correlation_threshold: Correlation threshold for redundant features
    
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    removed_cols = set()
    
    if remove_constant:
        constant_cols = filter_constant_features(filtered_df)
        removed_cols.update(constant_cols)
    
    if remove_low_variance:
        low_var_cols = filter_low_variance(filtered_df, threshold=variance_threshold)
        removed_cols.update(low_var_cols)
    
    if removed_cols:
        filtered_df = filtered_df.drop(columns=list(removed_cols))
        logger.info(f"Removed {len(removed_cols)} filtered columns. Remaining: {len(filtered_df.columns)}")
    
    if remove_highly_correlated and len(filtered_df.columns) > 1:
        redundant_cols = filter_highly_correlated(
            filtered_df,
            threshold=correlation_threshold
        )
        if redundant_cols:
            filtered_df = filtered_df.drop(columns=redundant_cols)
            logger.info(f"Removed {len(redundant_cols)} redundant columns. Remaining: {len(filtered_df.columns)}")
    
    return filtered_df
