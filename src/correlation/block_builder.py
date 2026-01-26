"""
Privacy block generation.
Groups correlated features into blocks for joint sensitivity handling.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Set, Dict, Optional
import logging

from .approx_corr import pairwise_correlations, approximate_correlation_matrix
from .mi_estimation import pairwise_mutual_information
from .prune_pairs import prune_correlation_pairs, prune_mi_pairs, combine_correlation_sources
from .graph_builder import build_dependency_graph, get_connected_components, get_communities

logger = logging.getLogger(__name__)


class PrivacyBlockBuilder:
    """Builds privacy blocks from feature dependencies."""
    
    def __init__(
        self,
        corr_threshold: float = 0.3,
        mi_threshold: float = 0.1,
        use_mi: bool = True,
        use_communities: bool = False,
        min_block_size: int = 1,
        max_block_size: Optional[int] = None
    ):
        """
        Initialize block builder.
        
        Args:
            corr_threshold: Minimum correlation to consider
            mi_threshold: Minimum MI to consider
            use_mi: Whether to use mutual information in addition to correlation
            use_communities: Use community detection instead of connected components
            min_block_size: Minimum features per block
            max_block_size: Maximum features per block (None = no limit)
        """
        self.corr_threshold = corr_threshold
        self.mi_threshold = mi_threshold
        self.use_mi = use_mi
        self.use_communities = use_communities
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.blocks: List[Set[str]] = []
    
    def build_blocks(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> List[Set[str]]:
        """
        Build privacy blocks from DataFrame.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to include (None = all numeric)
            sample_size: Sample size for correlation computation
        
        Returns:
            List of sets, each representing a privacy block
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Building privacy blocks for {len(columns)} columns")
        
        # Compute correlations
        logger.info("Computing Pearson correlations...")
        corr_pairs = pairwise_correlations(
            df,
            columns=columns,
            threshold=self.corr_threshold
        )
        corr_pairs = prune_correlation_pairs(corr_pairs, threshold=self.corr_threshold)
        
        # Compute MI if requested
        mi_pairs = []
        if self.use_mi:
            logger.info("Computing Mutual Information...")
            mi_pairs = pairwise_mutual_information(
                df,
                columns=columns,
                threshold=self.mi_threshold
            )
            mi_pairs = prune_mi_pairs(mi_pairs, threshold=self.mi_threshold)
        
        # Combine or use separately
        if self.use_mi and mi_pairs:
            logger.info("Combining correlation and MI...")
            dependency_pairs = combine_correlation_sources(corr_pairs, mi_pairs)
        else:
            dependency_pairs = [(c1, c2, abs(corr)) for c1, c2, corr in corr_pairs]
        
        # Build dependency graph
        logger.info("Building dependency graph...")
        graph = build_dependency_graph(dependency_pairs, threshold=0.0)
        
        # Extract blocks (connected components or communities)
        if self.use_communities:
            logger.info("Detecting communities...")
            blocks = get_communities(graph)
        else:
            logger.info("Finding connected components...")
            blocks = get_connected_components(graph)
        
        # Add isolated features as single-feature blocks
        all_features_in_blocks = set()
        for block in blocks:
            all_features_in_blocks.update(block)
        
        isolated_features = set(columns) - all_features_in_blocks
        for feature in isolated_features:
            blocks.append({feature})
        
        # Filter blocks by size
        filtered_blocks = []
        for block in blocks:
            if len(block) >= self.min_block_size:
                if self.max_block_size is None or len(block) <= self.max_block_size:
                    filtered_blocks.append(block)
                else:
                    # Split large blocks (simple strategy: take first max_block_size)
                    logger.warning(f"Splitting large block of size {len(block)}")
                    block_list = list(block)
                    for i in range(0, len(block_list), self.max_block_size):
                        filtered_blocks.append(set(block_list[i:i + self.max_block_size]))
        
        self.blocks = filtered_blocks
        logger.info(f"Created {len(self.blocks)} privacy blocks")
        
        # Log block sizes
        block_sizes = [len(block) for block in self.blocks]
        logger.info(f"Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, mean={sum(block_sizes)/len(block_sizes):.2f}")
        
        return self.blocks
    
    def save_blocks(self, filepath: str) -> None:
        """Save blocks to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for JSON serialization
        blocks_dict = {
            f"block_{i}": list(block)
            for i, block in enumerate(self.blocks)
        }
        
        with open(filepath, 'w') as f:
            json.dump(blocks_dict, f, indent=2)
        
        logger.info(f"Saved {len(self.blocks)} blocks to {filepath}")
    
    @classmethod
    def load_blocks(cls, filepath: str) -> List[Set[str]]:
        """Load blocks from JSON file."""
        with open(filepath, 'r') as f:
            blocks_dict = json.load(f)
        
        blocks = [set(block) for block in blocks_dict.values()]
        logger.info(f"Loaded {len(blocks)} blocks from {filepath}")
        return blocks


def build_privacy_blocks(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    corr_threshold: float = 0.3,
    mi_threshold: float = 0.1,
    **kwargs
) -> List[Set[str]]:
    """
    Convenience function to build and optionally save privacy blocks.
    
    Args:
        df: DataFrame to analyze
        output_path: Optional path to save blocks JSON
        corr_threshold: Correlation threshold
        mi_threshold: MI threshold
        **kwargs: Additional arguments for PrivacyBlockBuilder
    
    Returns:
        List of privacy blocks
    """
    builder = PrivacyBlockBuilder(
        corr_threshold=corr_threshold,
        mi_threshold=mi_threshold,
        **kwargs
    )
    
    blocks = builder.build_blocks(df)
    
    if output_path:
        builder.save_blocks(output_path)
    
    return blocks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build privacy blocks from dataset")
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--output", required=True, help="Output JSON file for blocks")
    parser.add_argument("--corr-threshold", type=float, default=0.3, help="Correlation threshold")
    parser.add_argument("--mi-threshold", type=float, default=0.1, help="MI threshold")
    parser.add_argument("--use-mi", action='store_true', help="Use mutual information")
    parser.add_argument("--use-communities", action='store_true', help="Use community detection")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    from ..ingestion.load_data import load_parquet
    df = load_parquet(args.input)
    
    # Build blocks
    blocks = build_privacy_blocks(
        df,
        output_path=args.output,
        corr_threshold=args.corr_threshold,
        mi_threshold=args.mi_threshold,
        use_mi=args.use_mi,
        use_communities=args.use_communities
    )
    
    print(f"Created {len(blocks)} privacy blocks")
