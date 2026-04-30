"""
Block-level sensitivity estimation.
Computes sensitivity for privacy blocks (required for DP mechanisms).

Key concepts:
- L1 sensitivity (for Laplace): sum of per-feature ranges in the block
- L2 sensitivity (for Gaussian): sqrt(sum of squared per-feature ranges)
- Per-feature sensitivity: individual feature range (max - min)
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_feature_sensitivity(
    bounds: Tuple[float, float]
) -> float:
    """
    Compute sensitivity for a single feature.

    For bounded features, sensitivity = max - min (the range).

    Args:
        bounds: (min, max) bounds for the feature

    Returns:
        Sensitivity (max - min)
    """
    min_val, max_val = bounds
    return max_val - min_val


def compute_per_feature_sensitivities(
    block: Set[str],
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """
    Compute per-feature sensitivities for all features in a block.

    Args:
        block: Set of feature names in the block
        bounds: Dict mapping feature names to (min, max) bounds

    Returns:
        Dict mapping feature name to its individual sensitivity (range)
    """
    sensitivities = {}
    for feature in block:
        if feature in bounds:
            min_val, max_val = bounds[feature]
            sensitivities[feature] = max_val - min_val
        else:
            logger.warning(f"No bounds for feature '{feature}', skipping")
    return sensitivities


def compute_block_l1_sensitivity(
    block: Set[str],
    bounds: Dict[str, Tuple[float, float]]
) -> float:
    """
    Compute L1 (sum) sensitivity for a privacy block.
    Used for Laplace mechanism under sequential composition.

    L1 sensitivity = sum of individual feature ranges.

    Args:
        block: Set of feature names in the block
        bounds: Dict mapping feature names to (min, max) bounds

    Returns:
        L1 sensitivity for the block
    """
    per_feature = compute_per_feature_sensitivities(block, bounds)
    if not per_feature:
        return 0.0
    return sum(per_feature.values())


def compute_block_l2_sensitivity(
    block: Set[str],
    bounds: Dict[str, Tuple[float, float]]
) -> float:
    """
    Compute L2 (Euclidean) sensitivity for a privacy block.
    Used for Gaussian mechanism.

    L2 sensitivity = sqrt(sum of squared individual feature ranges).

    Args:
        block: Set of feature names in the block
        bounds: Dict mapping feature names to (min, max) bounds

    Returns:
        L2 sensitivity for the block
    """
    per_feature = compute_per_feature_sensitivities(block, bounds)
    if not per_feature:
        return 0.0
    return float(np.sqrt(sum(s ** 2 for s in per_feature.values())))


def compute_block_sensitivity(
    df: pd.DataFrame,
    block: Set[str],
    bounds: Dict[str, Tuple[float, float]],
    mechanism: str = 'laplace'
) -> float:
    """
    Compute joint sensitivity for a privacy block.

    For Laplace: uses L1 sensitivity (sum of ranges)
    For Gaussian: uses L2 sensitivity (sqrt of sum of squared ranges)

    Args:
        df: DataFrame containing the features
        block: Set of feature names in the block
        bounds: Dict mapping feature names to (min, max) bounds
        mechanism: 'laplace' or 'gaussian'

    Returns:
        Joint sensitivity for the block
    """
    # Check all features in block have bounds
    missing_bounds = block - set(bounds.keys())
    if missing_bounds:
        raise ValueError(f"Missing bounds for features: {missing_bounds}")

    if mechanism == 'laplace':
        return compute_block_l1_sensitivity(block, bounds)
    elif mechanism == 'gaussian':
        return compute_block_l2_sensitivity(block, bounds)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")


def compute_all_block_sensitivities(
    blocks: List[Set[str]],
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
    mechanism: str = 'laplace'
) -> Dict[int, float]:
    """
    Compute sensitivities for all privacy blocks.

    Args:
        blocks: List of privacy blocks (sets of feature names)
        df: DataFrame containing the features
        bounds: Dict mapping feature names to (min, max) bounds
        mechanism: 'laplace' or 'gaussian'

    Returns:
        Dict mapping block index to sensitivity
    """
    sensitivities = {}

    for i, block in enumerate(blocks):
        sensitivity = compute_block_sensitivity(df, block, bounds, mechanism=mechanism)
        sensitivities[i] = sensitivity
        logger.debug(f"Block {i} (size={len(block)}): sensitivity={sensitivity:.4f}")

    logger.info(f"Computed sensitivities for {len(sensitivities)} blocks")
    return sensitivities


def get_block_features(blocks: List[Set[str]]) -> Dict[int, Set[str]]:
    """
    Get mapping from block index to feature set.

    Args:
        blocks: List of privacy blocks

    Returns:
        Dict mapping block index to feature set
    """
    return {i: block for i, block in enumerate(blocks)}
