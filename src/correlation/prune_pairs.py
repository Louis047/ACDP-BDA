"""
Threshold-based pruning of correlation pairs.
Filters weak correlations to focus on significant dependencies.
"""

import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def prune_correlation_pairs(
    pairs: List[Tuple[str, str, float]],
    threshold: float = 0.3,
    use_absolute: bool = True
) -> List[Tuple[str, str, float]]:
    """
    Prune correlation pairs below threshold.
    
    Args:
        pairs: List of (col1, col2, correlation) tuples
        threshold: Minimum correlation to keep
        use_absolute: If True, use absolute value of correlation
    
    Returns:
        Filtered list of pairs
    """
    if use_absolute:
        filtered = [
            (col1, col2, corr)
            for col1, col2, corr in pairs
            if abs(corr) >= threshold
        ]
    else:
        filtered = [
            (col1, col2, corr)
            for col1, col2, corr in pairs
            if corr >= threshold
        ]
    
    logger.info(f"Pruned {len(pairs)} pairs to {len(filtered)} pairs (threshold={threshold})")
    return filtered


def prune_mi_pairs(
    pairs: List[Tuple[str, str, float]],
    threshold: float = 0.1
) -> List[Tuple[str, str, float]]:
    """
    Prune mutual information pairs below threshold.
    
    Args:
        pairs: List of (col1, col2, MI) tuples
        threshold: Minimum MI to keep
    
    Returns:
        Filtered list of pairs
    """
    filtered = [
        (col1, col2, mi)
        for col1, col2, mi in pairs
        if mi >= threshold
    ]
    
    logger.info(f"Pruned {len(pairs)} MI pairs to {len(filtered)} pairs (threshold={threshold})")
    return filtered


def combine_correlation_sources(
    corr_pairs: List[Tuple[str, str, float]],
    mi_pairs: List[Tuple[str, str, float]],
    corr_weight: float = 0.5,
    mi_weight: float = 0.5,
    normalize: bool = True
) -> List[Tuple[str, str, float]]:
    """
    Combine Pearson correlation and MI into unified dependency scores.
    
    Args:
        corr_pairs: List of (col1, col2, correlation) tuples
        mi_pairs: List of (col1, col2, MI) tuples
        corr_weight: Weight for correlation component
        mi_weight: Weight for MI component
        normalize: Whether to normalize scores to [0, 1]
    
    Returns:
        List of (col1, col2, combined_score) tuples
    """
    # Create dictionaries for fast lookup
    corr_dict = {(min(c1, c2), max(c1, c2)): abs(corr) for c1, c2, corr in corr_pairs}
    mi_dict = {(min(c1, c2), max(c1, c2)): mi for c1, c2, mi in mi_pairs}
    
    # Get all unique pairs
    all_pairs = set(corr_dict.keys()) | set(mi_dict.keys())
    
    # Normalize if requested
    if normalize:
        if corr_dict:
            max_corr = max(corr_dict.values())
            if max_corr > 0:
                corr_dict = {k: v / max_corr for k, v in corr_dict.items()}
        
        if mi_dict:
            max_mi = max(mi_dict.values())
            if max_mi > 0:
                mi_dict = {k: v / max_mi for k, v in mi_dict.items()}
    
    # Combine scores
    combined_pairs = []
    for pair in all_pairs:
        corr_score = corr_dict.get(pair, 0.0) * corr_weight
        mi_score = mi_dict.get(pair, 0.0) * mi_weight
        combined_score = corr_score + mi_score
        
        combined_pairs.append((pair[0], pair[1], combined_score))
    
    # Sort by score (descending)
    combined_pairs.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Combined {len(corr_pairs)} correlation pairs and {len(mi_pairs)} MI pairs into {len(combined_pairs)} unified pairs")
    return combined_pairs
