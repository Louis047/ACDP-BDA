"""
Block-level joint sensitivity estimation.
Computes sensitivity for privacy blocks (required for DP mechanisms).
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_feature_sensitivity(
    values: np.ndarray,
    bounds: Tuple[float, float]
) -> float:
    """
    Compute sensitivity for a single feature.
    
    For DP, sensitivity is the maximum change in output when one row changes.
    For bounded features, sensitivity = max - min (the range).
    
    Args:
        values: Feature values
        bounds: (min, max) bounds for the feature
    
    Returns:
        Sensitivity (max - min)
    """
    min_val, max_val = bounds
    return max_val - min_val


def compute_block_sensitivity(
    df: pd.DataFrame,
    block: Set[str],
    bounds: Dict[str, Tuple[float, float]]
) -> float:
    """
    Compute joint sensitivity for a privacy block.
    
    For correlated features in a block, joint sensitivity accounts for
    the maximum possible change when one row is modified.
    
    This is a simplified version. In practice, joint sensitivity depends on
    the specific query/statistic being computed. For mean queries on bounded
    features, we use the L2 norm of individual sensitivities.
    
    Args:
        df: DataFrame containing the features
        block: Set of feature names in the block
        bounds: Dict mapping feature names to (min, max) bounds
    
    Returns:
        Joint sensitivity for the block
    """
    # Check all features in block have bounds
    missing_bounds = block - set(bounds.keys())
    if missing_bounds:
        raise ValueError(f"Missing bounds for features: {missing_bounds}")
    
    # Compute individual sensitivities
    individual_sensitivities = []
    for feature in block:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not in DataFrame")
            continue
        
        min_val, max_val = bounds[feature]
        sensitivity = max_val - min_val
        individual_sensitivities.append(sensitivity)
    
    if not individual_sensitivities:
        return 0.0
    
    # For joint sensitivity, use L2 norm (conservative bound)
    # This accounts for correlated changes across features
    joint_sensitivity = np.sqrt(sum(s**2 for s in individual_sensitivities))
    
    return float(joint_sensitivity)


def compute_all_block_sensitivities(
    blocks: List[Set[str]],
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]]
) -> Dict[int, float]:
    """
    Compute sensitivities for all privacy blocks.
    
    Args:
        blocks: List of privacy blocks (sets of feature names)
        df: DataFrame containing the features
        bounds: Dict mapping feature names to (min, max) bounds
    
    Returns:
        Dict mapping block index to sensitivity
    """
    sensitivities = {}
    
    for i, block in enumerate(blocks):
        sensitivity = compute_block_sensitivity(df, block, bounds)
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
