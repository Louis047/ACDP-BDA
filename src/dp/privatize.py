"""
Apply DP noise to dataset using block-level allocations.
Main entry point for privatizing entire datasets.

CRITICAL: For Laplace mechanism on individual features within a block:
- Each feature uses its OWN L1 sensitivity (range = max - min)
- The block's epsilon budget is SPLIT across features: ε_per_feature = ε_block / k
  where k = number of features in the block
- This satisfies sequential composition: total block privacy = sum of per-feature ε = ε_block
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple
import logging

from .sensitivity import (
    compute_all_block_sensitivities,
    compute_per_feature_sensitivities,
    compute_block_l2_sensitivity,
)
from .budget_allocator import AdaptiveBudgetAllocator
from .mechanisms import add_laplace_noise, add_gaussian_noise

logger = logging.getLogger(__name__)


class DatasetPrivatizer:
    """Privatizes datasets using block-level DP with correct noise calibration."""

    def __init__(
        self,
        blocks: List[Set[str]],
        bounds: Dict[str, tuple],
        total_epsilon: float,
        mechanism: str = 'laplace',
        allocation_method: str = 'optimal',
        delta: float = 1e-5,
        random_state: Optional[int] = None
    ):
        """
        Initialize privatizer.

        Args:
            blocks: List of privacy blocks (sets of feature names)
            bounds: Dict mapping feature names to (min, max) bounds
            total_epsilon: Total privacy budget (ε)
            mechanism: 'laplace' or 'gaussian'
            allocation_method: Budget allocation method
            delta: Privacy parameter for Gaussian mechanism
            random_state: Random seed for reproducibility
        """
        self.blocks = blocks
        self.bounds = bounds
        self.total_epsilon = total_epsilon
        self.mechanism = mechanism
        self.allocation_method = allocation_method
        self.delta = delta
        self.random_state = random_state

        self.sensitivities: Dict[int, float] = {}
        self.allocations: Dict[int, float] = {}

    def compute_sensitivities(self, df: pd.DataFrame) -> Dict[int, float]:
        """Compute sensitivities for all blocks."""
        self.sensitivities = compute_all_block_sensitivities(
            self.blocks,
            df,
            self.bounds,
            mechanism=self.mechanism
        )
        return self.sensitivities

    def allocate_budget(self) -> Dict[int, float]:
        """Allocate privacy budget across blocks."""
        if not self.sensitivities:
            raise ValueError("Must compute sensitivities before allocating budget")

        allocator = AdaptiveBudgetAllocator(
            method=self.allocation_method,
            blocks=self.blocks,
            bounds=self.bounds,
        )
        self.allocations = allocator.allocate(self.sensitivities, self.total_epsilon)
        return self.allocations

    def privatize_block(
        self,
        df: pd.DataFrame,
        block: Set[str],
        block_index: int
    ) -> pd.DataFrame:
        """
        Privatize features in a single block with CORRECT noise calibration.

        For Laplace mechanism:
            - Each feature gets its own L1 sensitivity (range)
            - Block epsilon is split: ε_per_feature = ε_block / n_features
            - Noise scale for feature i = Δ_i / ε_per_feature

        For Gaussian mechanism:
            - Uses block L2 sensitivity
            - All features share the same noise scale

        Args:
            df: DataFrame to privatize
            block: Set of feature names in the block
            block_index: Index of the block

        Returns:
            DataFrame with privatized features
        """
        if block_index not in self.allocations:
            raise ValueError(f"No budget allocated for block {block_index}")

        epsilon_block = self.allocations[block_index]

        # Get features that actually exist in the DataFrame
        valid_features = [f for f in block if f in df.columns]
        if not valid_features:
            logger.warning(f"No valid features in block {block_index}")
            return df.copy()

        n_features = len(valid_features)
        privatized_df = df.copy()

        # Generate unique RNG for this block (avoid correlated noise)
        block_seed = None
        if self.random_state is not None:
            block_seed = self.random_state + block_index * 1000

        if self.mechanism == 'laplace':
            # === CORRECT LAPLACE: per-feature sensitivity + split epsilon ===
            # By sequential composition: ε_block = sum(ε_per_feature)
            # So ε_per_feature = ε_block / n_features
            epsilon_per_feature = epsilon_block / n_features

            for feat_idx, feature in enumerate(valid_features):
                # Each feature uses its OWN sensitivity (range)
                min_val, max_val = self.bounds[feature]
                feature_sensitivity = max_val - min_val

                values = df[feature].values.astype(float)

                # Unique seed per feature within block
                feat_seed = None
                if block_seed is not None:
                    feat_seed = block_seed + feat_idx

                # Noise scale = Δ_feature / ε_per_feature
                privatized_values = add_laplace_noise(
                    values,
                    feature_sensitivity,
                    epsilon_per_feature,
                    random_state=feat_seed
                )

                # Clip to bounds
                privatized_values = np.clip(privatized_values, min_val, max_val)
                privatized_df[feature] = privatized_values

        elif self.mechanism == 'gaussian':
            # === GAUSSIAN: block L2 sensitivity, shared noise ===
            l2_sensitivity = compute_block_l2_sensitivity(block, self.bounds)

            for feat_idx, feature in enumerate(valid_features):
                min_val, max_val = self.bounds[feature]
                values = df[feature].values.astype(float)

                feat_seed = None
                if block_seed is not None:
                    feat_seed = block_seed + feat_idx

                privatized_values = add_gaussian_noise(
                    values,
                    l2_sensitivity,
                    epsilon_block,
                    delta=self.delta,
                    random_state=feat_seed
                )

                privatized_values = np.clip(privatized_values, min_val, max_val)
                privatized_df[feature] = privatized_values
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")

        return privatized_df

    def privatize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Privatize entire dataset.

        Args:
            df: DataFrame to privatize

        Returns:
            Privatized DataFrame
        """
        logger.info(f"Privatizing dataset with {len(self.blocks)} blocks, ε={self.total_epsilon:.4f}")

        # Compute sensitivities if not already done
        if not self.sensitivities:
            logger.info("Computing block sensitivities...")
            self.compute_sensitivities(df)

        # Allocate budget if not already done
        if not self.allocations:
            logger.info("Allocating privacy budget...")
            self.allocate_budget()

        # Log allocation summary
        for i, block in enumerate(self.blocks):
            eps = self.allocations.get(i, 0)
            logger.info(f"  Block {i} ({len(block)} features): ε={eps:.4f}")

        # Privatize each block
        privatized_df = df.copy()

        for i, block in enumerate(self.blocks):
            logger.debug(f"Privatizing block {i} ({len(block)} features)")
            block_df = self.privatize_block(df, block, i)

            # Update privatized dataframe
            for feature in block:
                if feature in block_df.columns:
                    privatized_df[feature] = block_df[feature]

        logger.info("Dataset privatization complete")
        return privatized_df

    def get_noise_report(self) -> Dict:
        """Get a report of noise scales used per feature."""
        report = {}
        for i, block in enumerate(self.blocks):
            eps_block = self.allocations.get(i, 0)
            valid_features = [f for f in block if f in self.bounds]
            n_features = len(valid_features)

            if self.mechanism == 'laplace' and n_features > 0:
                eps_per_feat = eps_block / n_features
                for feature in valid_features:
                    min_val, max_val = self.bounds[feature]
                    sensitivity = max_val - min_val
                    noise_scale = sensitivity / eps_per_feat if eps_per_feat > 0 else float('inf')
                    report[feature] = {
                        'block': i,
                        'sensitivity': sensitivity,
                        'epsilon': eps_per_feat,
                        'noise_scale': noise_scale,
                    }
        return report


def privatize_dataset(
    df: pd.DataFrame,
    blocks: List[Set[str]],
    bounds: Dict[str, tuple],
    total_epsilon: float,
    mechanism: str = 'laplace',
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to privatize a dataset.

    Args:
        df: DataFrame to privatize
        blocks: List of privacy blocks
        bounds: Feature bounds dict
        total_epsilon: Total privacy budget
        mechanism: Noise mechanism
        **kwargs: Additional arguments for DatasetPrivatizer

    Returns:
        Privatized DataFrame
    """
    privatizer = DatasetPrivatizer(
        blocks=blocks,
        bounds=bounds,
        total_epsilon=total_epsilon,
        mechanism=mechanism,
        **kwargs
    )
    return privatizer.privatize(df)
