"""
Feature-independent DP baseline.
Implements standard DP without correlation awareness for comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from ..dp.mechanisms import add_laplace_noise, add_gaussian_noise
from ..ingestion.bounds import BoundsEnforcer

logger = logging.getLogger(__name__)


def baseline_privatize(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    total_epsilon: float,
    mechanism: str = 'laplace',
    delta: float = 1e-5,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply feature-independent DP (baseline).
    
    Each feature gets equal share of privacy budget: ε_per_feature = ε / n_features
    
    Args:
        df: DataFrame to privatize
        bounds: Dict mapping feature names to (min, max) bounds
        total_epsilon: Total privacy budget
        mechanism: 'laplace' or 'gaussian'
        delta: Privacy parameter for Gaussian mechanism
        random_state: Random seed
    
    Returns:
        Privatized DataFrame
    """
    # Select numeric columns that have bounds
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bounded_cols = [col for col in numeric_cols if col in bounds]
    
    if not bounded_cols:
        logger.warning("No bounded numeric columns found")
        return df.copy()
    
    n_features = len(bounded_cols)
    epsilon_per_feature = total_epsilon / n_features
    
    logger.info(
        f"Baseline DP: allocating ε={epsilon_per_feature:.6f} per feature "
        f"({n_features} features, total ε={total_epsilon:.6f})"
    )
    
    privatized_df = df.copy()
    
    for col in bounded_cols:
        min_val, max_val = bounds[col]
        sensitivity = max_val - min_val
        
        values = df[col].values.astype(float)
        
        # Apply noise
        if mechanism == 'laplace':
            privatized_values = add_laplace_noise(
                values,
                sensitivity,
                epsilon_per_feature,
                random_state=random_state
            )
        elif mechanism == 'gaussian':
            privatized_values = add_gaussian_noise(
                values,
                sensitivity,
                epsilon_per_feature,
                delta=delta,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        # Clip to bounds
        privatized_values = np.clip(privatized_values, min_val, max_val)
        privatized_df[col] = privatized_values
    
    return privatized_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply baseline (feature-independent) DP")
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--bounds", required=True, help="Feature bounds JSON file")
    parser.add_argument("--epsilon", type=float, required=True, help="Total privacy budget")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--mechanism", choices=['laplace', 'gaussian'], default='laplace')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    from ..ingestion.load_data import load_parquet
    df = load_parquet(args.input)
    
    # Load bounds
    bounds_enforcer = BoundsEnforcer.load_bounds(args.bounds)
    bounds = bounds_enforcer.bounds
    
    # Apply baseline DP
    privatized_df = baseline_privatize(
        df,
        bounds,
        args.epsilon,
        mechanism=args.mechanism
    )
    
    # Save
    from ..ingestion.load_data import save_parquet
    save_parquet(privatized_df, args.output)
    
    print(f"Baseline privatized dataset saved to {args.output}")
