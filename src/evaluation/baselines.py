"""
Baseline DP implementations for comparison.
1. Feature-Independent (Uniform): standard per-feature DP with equal ε per feature
2. Random Blocking: randomly group features into blocks (no correlation awareness)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Set
import logging

from ..dp.mechanisms import add_laplace_noise, add_gaussian_noise

logger = logging.getLogger(__name__)


def baseline_uniform_privatize(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    total_epsilon: float,
    mechanism: str = 'laplace',
    delta: float = 1e-5,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply feature-independent DP (uniform baseline).

    Each feature gets equal share of privacy budget: ε_per_feature = ε / n_features
    Each feature uses its OWN L1 sensitivity (range).

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
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bounded_cols = [col for col in numeric_cols if col in bounds]

    if not bounded_cols:
        logger.warning("No bounded numeric columns found")
        return df.copy()

    n_features = len(bounded_cols)
    epsilon_per_feature = total_epsilon / n_features

    logger.info(
        f"Baseline Uniform DP: ε={epsilon_per_feature:.6f} per feature "
        f"({n_features} features, total ε={total_epsilon:.6f})"
    )

    privatized_df = df.copy()

    for feat_idx, col in enumerate(bounded_cols):
        min_val, max_val = bounds[col]
        sensitivity = max_val - min_val

        values = df[col].values.astype(float)

        feat_seed = None
        if random_state is not None:
            feat_seed = random_state + feat_idx

        if mechanism == 'laplace':
            privatized_values = add_laplace_noise(
                values, sensitivity, epsilon_per_feature,
                random_state=feat_seed
            )
        elif mechanism == 'gaussian':
            privatized_values = add_gaussian_noise(
                values, sensitivity, epsilon_per_feature,
                delta=delta, random_state=feat_seed
            )
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        privatized_values = np.clip(privatized_values, min_val, max_val)
        privatized_df[col] = privatized_values

    return privatized_df


# Keep backward compatibility
baseline_privatize = baseline_uniform_privatize


def baseline_random_blocking_privatize(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    total_epsilon: float,
    n_blocks: Optional[int] = None,
    mechanism: str = 'laplace',
    delta: float = 1e-5,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Apply DP with RANDOM feature blocking (no correlation awareness).

    Features are randomly assigned to blocks. Budget is allocated
    inversely proportional to block sensitivity (same strategy as ACADP),
    but blocks are NOT correlation-aware.

    This baseline proves that ACADP's correlation-aware grouping matters.

    Args:
        df: DataFrame to privatize
        bounds: Dict mapping feature names to (min, max) bounds
        total_epsilon: Total privacy budget
        n_blocks: Number of random blocks (default: sqrt(n_features))
        mechanism: 'laplace' or 'gaussian'
        delta: Privacy parameter for Gaussian mechanism
        random_state: Random seed

    Returns:
        Privatized DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bounded_cols = [col for col in numeric_cols if col in bounds]

    if not bounded_cols:
        logger.warning("No bounded numeric columns found")
        return df.copy()

    rng = np.random.default_rng(random_state)

    # Default: create sqrt(n) blocks
    if n_blocks is None:
        n_blocks = max(2, int(np.sqrt(len(bounded_cols))))

    n_blocks = min(n_blocks, len(bounded_cols))

    # Randomly assign features to blocks
    shuffled_cols = list(bounded_cols)
    rng.shuffle(shuffled_cols)

    random_blocks: List[Set[str]] = [set() for _ in range(n_blocks)]
    for i, col in enumerate(shuffled_cols):
        random_blocks[i % n_blocks].add(col)

    logger.info(
        f"Random Blocking DP: {n_blocks} blocks, "
        f"sizes={[len(b) for b in random_blocks]}, total ε={total_epsilon:.6f}"
    )

    # Use the same privatization logic as ACADP but with random blocks
    from ..dp.privatize import DatasetPrivatizer

    privatizer = DatasetPrivatizer(
        blocks=random_blocks,
        bounds=bounds,
        total_epsilon=total_epsilon,
        mechanism=mechanism,
        allocation_method='inverse_sensitivity',
        delta=delta,
        random_state=random_state
    )

    return privatizer.privatize(df)


def run_all_baselines(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    total_epsilon: float,
    mechanism: str = 'laplace',
    random_state: Optional[int] = None,
    n_random_trials: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Run all baseline methods for comparison.

    Args:
        df: Original DataFrame
        bounds: Feature bounds
        total_epsilon: Privacy budget
        mechanism: DP mechanism
        random_state: Random seed
        n_random_trials: Number of random blocking trials to average

    Returns:
        Dict mapping baseline name to privatized DataFrame
    """
    results = {}

    # Uniform baseline
    logger.info("Running uniform baseline...")
    results['uniform'] = baseline_uniform_privatize(
        df, bounds, total_epsilon, mechanism=mechanism,
        random_state=random_state
    )

    # Random blocking (averaged over multiple trials)
    logger.info(f"Running random blocking baseline ({n_random_trials} trials)...")
    random_dfs = []
    for trial in range(n_random_trials):
        trial_seed = (random_state + trial * 100) if random_state is not None else None
        rdf = baseline_random_blocking_privatize(
            df, bounds, total_epsilon, mechanism=mechanism,
            random_state=trial_seed
        )
        random_dfs.append(rdf)

    # Average across trials for more stable comparison
    avg_random_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bounded_cols = [col for col in numeric_cols if col in bounds]

    for col in bounded_cols:
        col_values = np.stack([rdf[col].values for rdf in random_dfs], axis=0)
        avg_random_df[col] = col_values.mean(axis=0)

    results['random_blocking'] = avg_random_df

    return results
