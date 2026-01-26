"""
End-to-end ACADP pipeline orchestration.
Executes all four workflows sequentially.
"""

import argparse
import logging
from pathlib import Path
import json
from typing import Optional
import pandas as pd

from .ingestion.bounds import enforce_feature_bounds

# Workflow 1: Ingestion
from .ingestion.load_data import load_data, save_parquet
from .ingestion.schema import SchemaValidator
from .ingestion.preprocess import DataPreprocessor
from .ingestion.bounds import BoundsEnforcer
from .ingestion.stats import compute_basic_stats, save_stats

# Workflow 2: Correlation
from .correlation.block_builder import PrivacyBlockBuilder

# Workflow 3: DP
from .dp.privatize import DatasetPrivatizer

# Workflow 4: Evaluation
from .evaluation.baselines import baseline_privatize
from .evaluation.comparisons import compare_acadp_vs_baseline, save_comparison
from .evaluation.utility_metrics import compute_all_utility_metrics

logger = logging.getLogger(__name__)


class ACADPPipeline:
    """End-to-end ACADP pipeline."""
    
    def __init__(
        self,
        epsilon: float,
        corr_threshold: float = 0.3,
        mi_threshold: float = 0.1,
        mechanism: str = 'laplace',
        allocation_method: str = 'proportional',
        output_dir: str = 'output'
    ):
        """
        Initialize pipeline.
        
        Args:
            epsilon: Total privacy budget
            corr_threshold: Correlation threshold for block building
            mi_threshold: MI threshold for block building
            mechanism: DP mechanism ('laplace' or 'gaussian')
            allocation_method: Budget allocation method
            output_dir: Output directory for intermediate and final results
        """
        self.epsilon = epsilon
        self.corr_threshold = corr_threshold
        self.mi_threshold = mi_threshold
        self.mechanism = mechanism
        self.allocation_method = allocation_method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_workflow1_ingestion(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> tuple:
        """
        Workflow 1: Data Ingestion & Preprocessing.
        
        Args:
            input_path: Path to raw dataset
            output_path: Optional output path (defaults to output_dir/preprocessed.parquet)
        
        Returns:
            Tuple of (preprocessed_df, bounds_dict)
        """
        logger.info("=" * 60)
        logger.info("WORKFLOW 1: DATA INGESTION & PREPROCESSING")
        logger.info("=" * 60)
        
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_data(input_path)
        
        # Validate schema
        logger.info("Validating schema...")
        validator = SchemaValidator()
        schema = validator.infer_schema(df)
        df = validator.validate(df)
        
        # Preprocess
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(df, handle_missing=True, encode_categorical=True)
        
        # Enforce bounds
        logger.info("Enforcing feature bounds...")
        bounds_enforcer = BoundsEnforcer()
        bounds = bounds_enforcer.estimate_bounds(df)
        df, bounds = enforce_feature_bounds(df, bounds, estimate_if_missing=True)
        
        # Save bounds
        bounds_path = self.output_dir / 'bounds.json'
        bounds_enforcer.bounds = bounds
        bounds_enforcer.save_bounds(bounds_path)
        logger.info(f"Bounds saved to {bounds_path}")
        
        # Compute and save stats
        stats = compute_basic_stats(df)
        stats_path = self.output_dir / 'preprocessing_stats.json'
        save_stats(stats, stats_path)
        logger.info(f"Statistics saved to {stats_path}")
        
        # Save preprocessed data
        if output_path is None:
            output_path = self.output_dir / 'preprocessed.parquet'
        save_parquet(df, output_path)
        logger.info(f"Preprocessed data saved to {output_path}")
        
        logger.info("Workflow 1 complete")
        return df, bounds
    
    def run_workflow2_correlation(
        self,
        df,
        output_path: Optional[str] = None
    ) -> list:
        """
        Workflow 2: Correlation Detection & Feature Grouping.
        
        Args:
            df: Preprocessed DataFrame
            output_path: Optional output path for blocks JSON
        
        Returns:
            List of privacy blocks
        """
        logger.info("=" * 60)
        logger.info("WORKFLOW 2: CORRELATION DETECTION & FEATURE GROUPING")
        logger.info("=" * 60)
        
        # Build privacy blocks
        logger.info("Building privacy blocks...")
        builder = PrivacyBlockBuilder(
            corr_threshold=self.corr_threshold,
            mi_threshold=self.mi_threshold,
            use_mi=True
        )
        blocks = builder.build_blocks(df)
        
        # Save blocks
        if output_path is None:
            output_path = self.output_dir / 'privacy_blocks.json'
        builder.save_blocks(output_path)
        logger.info(f"Privacy blocks saved to {output_path}")
        
        logger.info("Workflow 2 complete")
        return blocks
    
    def run_workflow3_dp(
        self,
        df,
        blocks: list,
        bounds: dict,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Workflow 3: Differential Privacy & Budget Allocation.
        
        Args:
            df: Preprocessed DataFrame
            blocks: List of privacy blocks
            bounds: Feature bounds dict
            output_path: Optional output path for privatized data
        
        Returns:
            Privatized DataFrame
        """
        logger.info("=" * 60)
        logger.info("WORKFLOW 3: DIFFERENTIAL PRIVACY & BUDGET ALLOCATION")
        logger.info("=" * 60)
        
        # Privatize dataset
        logger.info(f"Privatizing dataset with ε={self.epsilon:.4f}")
        privatizer = DatasetPrivatizer(
            blocks=blocks,
            bounds=bounds,
            total_epsilon=self.epsilon,
            mechanism=self.mechanism,
            allocation_method=self.allocation_method
        )
        privatized_df = privatizer.privatize(df)
        
        # Save privatized data
        if output_path is None:
            output_path = self.output_dir / 'acadp_privatized.parquet'
        save_parquet(privatized_df, output_path)
        logger.info(f"Privatized data saved to {output_path}")
        
        logger.info("Workflow 3 complete")
        return privatized_df
    
    def run_workflow4_evaluation(
        self,
        original_df,
        acadp_df,
        bounds: dict,
        output_path: Optional[str] = None
    ) -> dict:
        """
        Workflow 4: Evaluation & Validation.
        
        Args:
            original_df: Original preprocessed DataFrame
            acadp_df: ACADP-privatized DataFrame
            bounds: Feature bounds dict
            output_path: Optional output path for comparison JSON
        
        Returns:
            Comparison dict
        """
        logger.info("=" * 60)
        logger.info("WORKFLOW 4: EVALUATION & VALIDATION")
        logger.info("=" * 60)
        
        # Generate baseline
        logger.info("Generating baseline (feature-independent DP)...")
        baseline_df = baseline_privatize(
            original_df,
            bounds,
            self.epsilon,
            mechanism=self.mechanism
        )
        baseline_path = self.output_dir / 'baseline_privatized.parquet'
        save_parquet(baseline_df, baseline_path)
        logger.info(f"Baseline privatized data saved to {baseline_path}")
        
        # Compare
        logger.info("Comparing ACADP vs baseline...")
        comparison = compare_acadp_vs_baseline(
            original_df,
            acadp_df,
            baseline_df
        )
        
        # Save comparison
        if output_path is None:
            output_path = self.output_dir / 'comparison.json'
        save_comparison(comparison, output_path)
        logger.info(f"Comparison saved to {output_path}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        if 'improvements' in comparison:
            for metric, value in comparison['improvements'].items():
                logger.info(f"{metric}: {value:.2f}%")
        logger.info("=" * 60)
        
        logger.info("Workflow 4 complete")
        return comparison
    
    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> dict:
        """
        Run complete ACADP pipeline.
        
        Args:
            input_path: Path to raw dataset
            output_path: Optional output path for final privatized data
        
        Returns:
            Comparison dict from evaluation
        """
        logger.info("=" * 60)
        logger.info("ACADP PIPELINE - STARTING")
        logger.info("=" * 60)
        logger.info(f"Privacy budget (ε): {self.epsilon:.4f}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Workflow 1: Ingestion
        df, bounds = self.run_workflow1_ingestion(input_path)
        
        # Workflow 2: Correlation
        blocks = self.run_workflow2_correlation(df)
        
        # Workflow 3: DP
        privatized_df = self.run_workflow3_dp(df, blocks, bounds, output_path)
        
        # Workflow 4: Evaluation
        comparison = self.run_workflow4_evaluation(df, privatized_df, bounds)
        
        logger.info("=" * 60)
        logger.info("ACADP PIPELINE - COMPLETE")
        logger.info("=" * 60)
        
        return comparison


def main():
    """Command-line interface for pipeline."""
    parser = argparse.ArgumentParser(description="ACADP Pipeline")
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--epsilon", type=float, required=True, help="Privacy budget (ε)")
    parser.add_argument("--output", help="Output path for privatized data")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--corr-threshold", type=float, default=0.3, help="Correlation threshold")
    parser.add_argument("--mi-threshold", type=float, default=0.1, help="MI threshold")
    parser.add_argument("--mechanism", choices=['laplace', 'gaussian'], default='laplace')
    parser.add_argument("--allocation-method", default='proportional')
    parser.add_argument("--log-level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    pipeline = ACADPPipeline(
        epsilon=args.epsilon,
        corr_threshold=args.corr_threshold,
        mi_threshold=args.mi_threshold,
        mechanism=args.mechanism,
        allocation_method=args.allocation_method,
        output_dir=args.output_dir
    )
    
    comparison = pipeline.run(args.input, args.output)
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
