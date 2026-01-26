"""
Apply DP noise to dataset using block-level allocations.
Main entry point for privatizing entire datasets.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Set, List, Optional
import logging

from .sensitivity import compute_all_block_sensitivities
from .budget_allocator import AdaptiveBudgetAllocator
from .mechanisms import add_laplace_noise, add_gaussian_noise
from ..ingestion.bounds import BoundsEnforcer

logger = logging.getLogger(__name__)


class DatasetPrivatizer:
    """Privatizes datasets using block-level DP."""
    
    def __init__(
        self,
        blocks: List[Set[str]],
        bounds: Dict[str, tuple],
        total_epsilon: float,
        mechanism: str = 'laplace',
        allocation_method: str = 'proportional',
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
            random_state: Random seed
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
        from .sensitivity import compute_all_block_sensitivities
        
        self.sensitivities = compute_all_block_sensitivities(
            self.blocks,
            df,
            self.bounds
        )
        return self.sensitivities
    
    def allocate_budget(self) -> Dict[int, float]:
        """Allocate privacy budget across blocks."""
        if not self.sensitivities:
            raise ValueError("Must compute sensitivities before allocating budget")
        
        allocator = AdaptiveBudgetAllocator(method=self.allocation_method)
        self.allocations = allocator.allocate(self.sensitivities, self.total_epsilon)
        return self.allocations
    
    def privatize_block(
        self,
        df: pd.DataFrame,
        block: Set[str],
        block_index: int
    ) -> pd.DataFrame:
        """
        Privatize features in a single block.
        
        Args:
            df: DataFrame to privatize
            block: Set of feature names in the block
            block_index: Index of the block
        
        Returns:
            DataFrame with privatized features (only this block's features)
        """
        if block_index not in self.allocations:
            raise ValueError(f"No budget allocated for block {block_index}")
        
        epsilon_block = self.allocations[block_index]
        sensitivity_block = self.sensitivities[block_index]
        
        privatized_df = df.copy()
        
        for feature in block:
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' not in DataFrame")
                continue
            
            values = df[feature].values.astype(float)
            
            # Apply noise mechanism
            if self.mechanism == 'laplace':
                privatized_values = add_laplace_noise(
                    values,
                    sensitivity_block,
                    epsilon_block,
                    random_state=self.random_state
                )
            elif self.mechanism == 'gaussian':
                privatized_values = add_gaussian_noise(
                    values,
                    sensitivity_block,
                    epsilon_block,
                    delta=self.delta,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unknown mechanism: {self.mechanism}")
            
            # Clip to bounds to ensure valid values
            min_val, max_val = self.bounds[feature]
            privatized_values = np.clip(privatized_values, min_val, max_val)
            
            privatized_df[feature] = privatized_values
        
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Privatize dataset using ACADP")
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--blocks", required=True, help="Privacy blocks JSON file")
    parser.add_argument("--bounds", required=True, help="Feature bounds JSON file")
    parser.add_argument("--epsilon", type=float, required=True, help="Total privacy budget (ε)")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--mechanism", choices=['laplace', 'gaussian'], default='laplace')
    parser.add_argument("--allocation-method", default='proportional')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    from ..ingestion.load_data import load_parquet
    df = load_parquet(args.input)
    
    # Load blocks
    with open(args.blocks, 'r') as f:
        blocks_dict = json.load(f)
    blocks = [set(block) for block in blocks_dict.values()]
    
    # Load bounds
    bounds_enforcer = BoundsEnforcer.load_bounds(args.bounds)
    bounds = bounds_enforcer.bounds
    
    # Privatize
    privatized_df = privatize_dataset(
        df,
        blocks,
        bounds,
        args.epsilon,
        mechanism=args.mechanism,
        allocation_method=args.allocation_method
    )
    
    # Save
    from ..ingestion.load_data import save_parquet
    save_parquet(privatized_df, args.output)
    
    print(f"Privatized dataset saved to {args.output}")
