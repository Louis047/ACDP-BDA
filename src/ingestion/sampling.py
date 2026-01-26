"""
Scalable sampling for correlation estimation.
Enables efficient correlation detection on large datasets.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def sample_dataframe(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Sample rows from DataFrame for efficient processing.
    
    Args:
        df: DataFrame to sample from
        n_samples: Number of samples to take
        fraction: Fraction of data to sample (0.0 to 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    if n_samples is None and fraction is None:
        raise ValueError("Either n_samples or fraction must be provided")
    
    if n_samples is not None and fraction is not None:
        raise ValueError("Provide either n_samples or fraction, not both")
    
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_samples = int(len(df) * fraction)
    
    if n_samples >= len(df):
        logger.info(f"Sample size ({n_samples}) >= dataset size ({len(df)}). Returning full dataset.")
        return df.copy()
    
    logger.info(f"Sampling {n_samples} rows from {len(df)} total rows")
    sampled = df.sample(n=n_samples, random_state=random_state)
    
    return sampled


def stratified_sample(
    df: pd.DataFrame,
    stratify_column: str,
    n_samples: Optional[int] = None,
    fraction: Optional[float] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Stratified sampling to preserve class distribution.
    
    Args:
        df: DataFrame to sample from
        stratify_column: Column to stratify on
        n_samples: Total number of samples
        fraction: Fraction of data to sample
        random_state: Random seed
    
    Returns:
        Stratified sampled DataFrame
    """
    if stratify_column not in df.columns:
        raise ValueError(f"Stratify column '{stratify_column}' not in DataFrame")
    
    if n_samples is None and fraction is None:
        raise ValueError("Either n_samples or fraction must be provided")
    
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        n_samples = int(len(df) * fraction)
    
    logger.info(f"Stratified sampling {n_samples} rows from {len(df)} total rows")
    
    # Use sklearn's stratified sampling
    from sklearn.model_selection import train_test_split
    
    _, sampled = train_test_split(
        df,
        train_size=n_samples,
        stratify=df[stratify_column],
        random_state=random_state
    )
    
    return sampled


def reservoir_sample(
    df: pd.DataFrame,
    k: int,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Reservoir sampling for streaming/chunked data.
    Maintains a random sample of k items from a stream.
    
    Args:
        df: DataFrame to sample from
        k: Sample size
        random_state: Random seed
    
    Returns:
        Sampled DataFrame
    """
    if k >= len(df):
        return df.copy()
    
    logger.info(f"Reservoir sampling {k} rows from {len(df)} total rows")
    
    rng = np.random.default_rng(random_state)
    reservoir = []
    
    for i, row in df.iterrows():
        if len(reservoir) < k:
            reservoir.append(row)
        else:
            j = rng.integers(0, i + 1)
            if j < k:
                reservoir[j] = row
    
    sampled_df = pd.DataFrame(reservoir)
    return sampled_df
