"""
End-to-end ACADP pipeline orchestration.
Executes all four workflows sequentially with comprehensive evaluation.
"""

import argparse
import logging
import time
import json
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

# Workflow 1: Ingestion
from .ingestion.load_data import load_data, save_parquet
from .ingestion.schema import SchemaValidator
from .ingestion.preprocess import DataPreprocessor
from .ingestion.bounds import BoundsEnforcer, enforce_feature_bounds
from .ingestion.stats import compute_basic_stats, save_stats

# Workflow 2: Correlation
from .correlation.block_builder import PrivacyBlockBuilder

# Workflow 3: DP
from .dp.privatize import DatasetPrivatizer

# Workflow 4: Evaluation
from .evaluation.baselines import (
    baseline_uniform_privatize,
    baseline_random_blocking_privatize,
    run_all_baselines,
)
from .evaluation.comparisons import (
    compare_three_way,
    run_multi_epsilon_sweep,
    save_comparison,
)
from .evaluation.utility_metrics import compute_all_utility_metrics
from .evaluation.downstream_ml import evaluate_downstream_ml
from .evaluation.plots import generate_all_plots

logger = logging.getLogger(__name__)


class ACADPPipeline:
    """End-to-end ACADP pipeline with comprehensive evaluation."""

    def __init__(
        self,
        epsilon: float,
        corr_threshold: float = 0.4,
        mi_threshold: float = 0.1,
        mechanism: str = 'laplace',
        allocation_method: str = 'optimal',
        output_dir: str = 'output',
        random_state: int = 42,
    ):
        self.epsilon = epsilon
        self.corr_threshold = corr_threshold
        self.mi_threshold = mi_threshold
        self.mechanism = mechanism
        self.allocation_method = allocation_method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Pipeline state
        self.df: Optional[pd.DataFrame] = None
        self.bounds: Optional[Dict] = None
        self.blocks: Optional[List] = None
        self.privatized_df: Optional[pd.DataFrame] = None
        self.privatizer: Optional[DatasetPrivatizer] = None

    def run_workflow1_ingestion(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> tuple:
        """Workflow 1: Data Ingestion & Preprocessing."""
        logger.info("=" * 60)
        logger.info("WORKFLOW 1: DATA INGESTION & PREPROCESSING")
        logger.info("=" * 60)

        start = time.time()

        # Load data
        logger.info(f"Loading data from {input_path}")
        df = load_data(input_path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Validate schema
        logger.info("Validating schema...")
        validator = SchemaValidator()
        schema = validator.infer_schema(df)
        validator.schema = schema
        df = validator.validate(df)

        # Preprocess
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        df = preprocessor.preprocess(df, handle_missing=True, encode_categorical=True)
        logger.info(f"After preprocessing: {len(df):,} rows, {len(df.columns)} columns")

        # Enforce bounds
        logger.info("Enforcing feature bounds...")
        df, bounds = enforce_feature_bounds(df, estimate_if_missing=True)

        # Save bounds
        bounds_enforcer = BoundsEnforcer(bounds)
        bounds_path = self.output_dir / 'bounds.json'
        bounds_enforcer.save_bounds(str(bounds_path))
        logger.info(f"Bounds saved to {bounds_path}")

        # Compute and save stats
        stats = compute_basic_stats(df)
        stats_path = self.output_dir / 'preprocessing_stats.json'
        save_stats(stats, str(stats_path))

        # Save preprocessed data
        if output_path is None:
            output_path = str(self.output_dir / 'preprocessed.parquet')
        save_parquet(df, output_path)

        elapsed = time.time() - start
        logger.info(f"Workflow 1 complete ({elapsed:.1f}s)")

        self.df = df
        self.bounds = bounds
        return df, bounds

    def run_workflow2_correlation(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = None
    ) -> list:
        """Workflow 2: Correlation Detection & Feature Grouping."""
        logger.info("=" * 60)
        logger.info("WORKFLOW 2: CORRELATION DETECTION & FEATURE GROUPING")
        logger.info("=" * 60)

        if df is None:
            df = self.df

        start = time.time()

        builder = PrivacyBlockBuilder(
            corr_threshold=self.corr_threshold,
            mi_threshold=self.mi_threshold,
            use_mi=True,
            use_communities=True,
            max_block_size=6,
        )
        blocks = builder.build_blocks(df)

        if output_path is None:
            output_path = str(self.output_dir / 'privacy_blocks.json')
        builder.save_blocks(output_path)

        # Log block composition
        for i, block in enumerate(blocks):
            logger.info(f"  Block {i}: {sorted(block)}")

        elapsed = time.time() - start
        logger.info(f"Workflow 2 complete ({elapsed:.1f}s)")

        self.blocks = blocks
        return blocks

    def run_workflow3_dp(
        self,
        df: Optional[pd.DataFrame] = None,
        blocks: Optional[list] = None,
        bounds: Optional[dict] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Workflow 3: Differential Privacy & Budget Allocation."""
        logger.info("=" * 60)
        logger.info("WORKFLOW 3: DIFFERENTIAL PRIVACY & BUDGET ALLOCATION")
        logger.info("=" * 60)

        if df is None:
            df = self.df
        if blocks is None:
            blocks = self.blocks
        if bounds is None:
            bounds = self.bounds

        start = time.time()

        logger.info(f"Privatizing with ε={self.epsilon:.4f}, "
                    f"mechanism={self.mechanism}, allocation={self.allocation_method}")

        privatizer = DatasetPrivatizer(
            blocks=blocks,
            bounds=bounds,
            total_epsilon=self.epsilon,
            mechanism=self.mechanism,
            allocation_method=self.allocation_method,
            random_state=self.random_state
        )
        privatized_df = privatizer.privatize(df)

        # Save
        if output_path is None:
            output_path = str(self.output_dir / 'acadp_privatized.parquet')
        save_parquet(privatized_df, output_path)

        # Save noise report
        noise_report = privatizer.get_noise_report()
        noise_report_path = self.output_dir / 'noise_report.json'
        with open(noise_report_path, 'w') as f:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in noise_report.items()}, f, indent=2)

        elapsed = time.time() - start
        logger.info(f"Workflow 3 complete ({elapsed:.1f}s)")

        self.privatized_df = privatized_df
        self.privatizer = privatizer
        return privatized_df

    def run_workflow4_evaluation(
        self,
        original_df: Optional[pd.DataFrame] = None,
        acadp_df: Optional[pd.DataFrame] = None,
        bounds: Optional[dict] = None,
        blocks: Optional[list] = None,
        run_sweep: bool = True,
        run_ml: bool = True,
        epsilon_values: Optional[List[float]] = None,
        n_trials: int = 5,
        classification_target: Optional[str] = None,
        regression_target: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> dict:
        """Workflow 4: Comprehensive Evaluation & Visualization."""
        logger.info("=" * 60)
        logger.info("WORKFLOW 4: EVALUATION & VALIDATION")
        logger.info("=" * 60)

        if original_df is None:
            original_df = self.df
        if acadp_df is None:
            acadp_df = self.privatized_df
        if bounds is None:
            bounds = self.bounds
        if blocks is None:
            blocks = self.blocks

        start = time.time()

        # Get numeric columns
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
        bounded_cols = [col for col in numeric_cols if col in bounds]

        # Generate baselines
        logger.info("Generating baselines...")
        baselines = run_all_baselines(
            original_df, bounds, self.epsilon,
            mechanism=self.mechanism,
            random_state=self.random_state,
            n_random_trials=n_trials
        )
        uniform_df = baselines['uniform']
        random_df = baselines['random_blocking']

        # Save baseline outputs
        save_parquet(uniform_df, str(self.output_dir / 'baseline_uniform.parquet'))
        save_parquet(random_df, str(self.output_dir / 'baseline_random.parquet'))

        # Three-way comparison
        logger.info("Computing three-way comparison...")
        comparison = compare_three_way(
            original_df, acadp_df, uniform_df, random_df,
            columns=bounded_cols
        )

        # Multi-ε sweep
        sweep_results = None
        if run_sweep:
            if epsilon_values is None:
                epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
            logger.info(f"Running multi-ε sweep: {epsilon_values}")
            sweep_results = run_multi_epsilon_sweep(
                original_df, blocks, bounds,
                epsilon_values=epsilon_values,
                mechanism=self.mechanism,
                allocation_method=self.allocation_method,
                random_state=self.random_state,
                n_trials=n_trials,
                columns=bounded_cols
            )

        # Downstream ML evaluation
        ml_results = None
        if run_ml:
            logger.info("Running downstream ML evaluation...")
            privatized_dfs = {
                'ACADP': acadp_df,
                'Uniform': uniform_df,
                'Random_Blocking': random_df
            }
            ml_results = evaluate_downstream_ml(
                original_df, privatized_dfs,
                classification_target=classification_target,
                regression_target=regression_target,
                feature_cols=bounded_cols,
                random_state=self.random_state
            )

        # Save results
        if output_path is None:
            output_path = str(self.output_dir / 'comparison.json')
        save_comparison(comparison, output_path)

        if sweep_results:
            save_comparison(sweep_results, str(self.output_dir / 'sweep_results.json'))

        if ml_results:
            save_comparison(ml_results, str(self.output_dir / 'ml_results.json'))

        # Generate plots
        logger.info("Generating plots...")
        plot_dir = str(self.output_dir / 'plots')

        allocations = self.privatizer.allocations if self.privatizer else {}

        generate_all_plots(
            original_df, acadp_df, uniform_df, random_df,
            comparison, blocks, allocations, self.epsilon,
            columns=bounded_cols,
            sweep_results=sweep_results,
            ml_results=ml_results,
            output_dir=plot_dir
        )

        # Print summary
        self._print_summary(comparison, ml_results)

        elapsed = time.time() - start
        logger.info(f"Workflow 4 complete ({elapsed:.1f}s)")

        return comparison

    def _print_summary(self, comparison: Dict, ml_results: Optional[Dict] = None):
        """Print evaluation summary to console."""
        print("\n" + "=" * 70)
        print("  ACADP EVALUATION SUMMARY")
        print("=" * 70)

        improvements = comparison.get('improvements_vs_uniform', {})
        if improvements:
            print("\n  ACADP Improvement vs Uniform Baseline:")
            print("  " + "-" * 50)
            metric_labels = {
                'mean_mse': 'Mean Squared Error',
                'mean_mae': 'Mean Absolute Error',
                'mean_relative_error': 'Relative Error',
                'correlation_error': 'Correlation Error',
                'mean_kl_divergence': 'KL Divergence',
                'mean_js_divergence': 'JS Divergence',
                'mean_wasserstein_distance': 'Wasserstein Distance',
                'mean_ks_statistic': 'KS Statistic',
                'mean_query_error': 'Query Error',
            }
            for key, label in metric_labels.items():
                if key in improvements:
                    val = improvements[key]
                    symbol = "✓" if val > 0 else "✗"
                    print(f"    {symbol} {label:.<35} {val:>+7.2f}%")

        improvements_random = comparison.get('improvements_vs_random', {})
        if improvements_random:
            print("\n  ACADP Improvement vs Random Blocking:")
            print("  " + "-" * 50)
            for key, label in metric_labels.items():
                if key in improvements_random:
                    val = improvements_random[key]
                    symbol = "✓" if val > 0 else "✗"
                    print(f"    {symbol} {label:.<35} {val:>+7.2f}%")

        if ml_results:
            print("\n  Downstream ML Performance:")
            print("  " + "-" * 50)
            for method, results in ml_results.items():
                if 'regression' in results and results['regression']:
                    r = results['regression']
                    print(f"    {method}: RMSE={r.get('privatized_rmse', 0):.4f}, "
                          f"R²={r.get('privatized_r2', 0):.4f}")

        print("\n" + "=" * 70)
        print(f"  Results saved to: {self.output_dir}")
        print(f"  Plots saved to: {self.output_dir / 'plots'}")
        print("=" * 70 + "\n")

    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        run_sweep: bool = True,
        run_ml: bool = True,
        epsilon_values: Optional[List[float]] = None,
        n_trials: int = 5,
        classification_target: Optional[str] = None,
        regression_target: Optional[str] = None,
    ) -> dict:
        """Run complete ACADP pipeline."""
        logger.info("=" * 60)
        logger.info("ACADP PIPELINE - STARTING")
        logger.info("=" * 60)
        logger.info(f"Privacy budget (ε): {self.epsilon:.4f}")
        logger.info(f"Mechanism: {self.mechanism}")
        logger.info(f"Allocation: {self.allocation_method}")
        logger.info(f"Output: {self.output_dir}")

        total_start = time.time()

        # Workflow 1
        df, bounds = self.run_workflow1_ingestion(input_path)

        # Workflow 2
        blocks = self.run_workflow2_correlation(df)

        # Workflow 3
        privatized_df = self.run_workflow3_dp(df, blocks, bounds, output_path)

        # Workflow 4
        comparison = self.run_workflow4_evaluation(
            df, privatized_df, bounds, blocks,
            run_sweep=run_sweep,
            run_ml=run_ml,
            epsilon_values=epsilon_values,
            n_trials=n_trials,
            classification_target=classification_target,
            regression_target=regression_target,
        )

        total_elapsed = time.time() - total_start
        logger.info(f"ACADP PIPELINE COMPLETE ({total_elapsed:.1f}s total)")

        return comparison


def main():
    """Command-line interface for pipeline."""
    parser = argparse.ArgumentParser(description="ACADP Pipeline")
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Privacy budget (ε)")
    parser.add_argument("--output", help="Output path for privatized data")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--corr-threshold", type=float, default=0.3)
    parser.add_argument("--mi-threshold", type=float, default=0.1)
    parser.add_argument("--mechanism", choices=['laplace', 'gaussian'], default='laplace')
    parser.add_argument("--allocation-method", default='inverse_sensitivity')
    parser.add_argument("--log-level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument("--no-sweep", action='store_true', help="Skip multi-ε sweep")
    parser.add_argument("--no-ml", action='store_true', help="Skip ML evaluation")
    parser.add_argument("--epsilon-values", nargs='+', type=float, default=None)
    parser.add_argument("--n-trials", type=int, default=5, help="Number of averaging trials")
    parser.add_argument("--classification-target", help="Column for classification task")
    parser.add_argument("--regression-target", help="Column for regression task")
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = ACADPPipeline(
        epsilon=args.epsilon,
        corr_threshold=args.corr_threshold,
        mi_threshold=args.mi_threshold,
        mechanism=args.mechanism,
        allocation_method=args.allocation_method,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )

    comparison = pipeline.run(
        args.input,
        args.output,
        run_sweep=not args.no_sweep,
        run_ml=not args.no_ml,
        epsilon_values=args.epsilon_values,
        n_trials=args.n_trials,
        classification_target=args.classification_target,
        regression_target=args.regression_target,
    )

    print("\nPipeline completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
