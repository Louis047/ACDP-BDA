"""
Comprehensive end-to-end test of the ACADP pipeline.
Runs all 4 workflows on NYC Taxi data with full evaluation.
"""

import sys
import logging
import time

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import pandas as pd
import numpy as np

# Suppress verbose warnings from sub-modules
logging.getLogger('src.correlation').setLevel(logging.WARNING)
logging.getLogger('src.dp').setLevel(logging.WARNING)
logging.getLogger('src.evaluation').setLevel(logging.INFO)

from src.ingestion.preprocess import DataPreprocessor
from src.ingestion.bounds import enforce_feature_bounds
from src.correlation.block_builder import PrivacyBlockBuilder
from src.dp.privatize import DatasetPrivatizer
from src.evaluation.baselines import baseline_uniform_privatize, baseline_random_blocking_privatize, run_all_baselines
from src.evaluation.comparisons import compare_three_way, run_multi_epsilon_sweep, save_comparison
from src.evaluation.utility_metrics import compute_all_utility_metrics
from src.evaluation.downstream_ml import evaluate_downstream_ml
from src.evaluation.plots import generate_all_plots

logger = logging.getLogger('main')


def main():
    total_start = time.time()

    # ========== CONFIGURATION ==========
    SAMPLE_SIZE = 200000
    EPSILON = 1.0
    MECHANISM = 'laplace'
    ALLOCATION_METHOD = 'optimal'
    RANDOM_STATE = 42
    OUTPUT_DIR = 'output'
    N_TRIALS = 3
    EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

    NUMERIC_COLS = [
        'passenger_count', 'trip_distance', 'fare_amount', 'extra',
        'mta_tax', 'tip_amount', 'tolls_amount', 'total_amount',
        'congestion_surcharge', 'airport_fee'
    ]

    # ========== WORKFLOW 1: INGESTION ==========
    logger.info("=" * 60)
    logger.info("WORKFLOW 1: DATA INGESTION & PREPROCESSING")
    logger.info("=" * 60)

    logger.info(f"Loading NYC Taxi data (sampling {SAMPLE_SIZE:,} rows)...")
    df = pd.read_parquet('data/yellow_tripdata_2023-01.parquet')
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    # Drop NaN in numeric columns
    df = df.dropna(subset=NUMERIC_COLS)
    logger.info(f"After cleanup: {len(df):,} rows, {len(NUMERIC_COLS)} features")

    # Enforce bounds
    bounds = {}
    for col in NUMERIC_COLS:
        col_min = max(0, float(df[col].quantile(0.01)))
        col_max = float(df[col].quantile(0.99))
        if col_max <= col_min:
            col_max = col_min + 1.0
        bounds[col] = (col_min, col_max)
        df[col] = df[col].clip(col_min, col_max)

    logger.info(f"Bounds: { {col: '%.2f-%.2f' % (b[0], b[1]) for col, b in bounds.items()} }")

    # ========== WORKFLOW 2: CORRELATION DETECTION ==========
    logger.info("=" * 60)
    logger.info("WORKFLOW 2: CORRELATION DETECTION & FEATURE GROUPING")
    logger.info("=" * 60)

    builder = PrivacyBlockBuilder(
        corr_threshold=0.4,
        mi_threshold=0.1,
        use_mi=True,
        use_communities=True,
        max_block_size=6,
    )
    blocks = builder.build_blocks(df, columns=NUMERIC_COLS)

    logger.info(f"Detected {len(blocks)} privacy blocks:")
    for i, b in enumerate(blocks):
        logger.info(f"  Block {i}: {sorted(b)}")

    # ========== WORKFLOW 3: DIFFERENTIAL PRIVACY ==========
    logger.info("=" * 60)
    logger.info("WORKFLOW 3: DIFFERENTIAL PRIVACY (ACADP)")
    logger.info("=" * 60)

    privatizer = DatasetPrivatizer(
        blocks=blocks,
        bounds=bounds,
        total_epsilon=EPSILON,
        mechanism=MECHANISM,
        allocation_method=ALLOCATION_METHOD,
        random_state=RANDOM_STATE,
    )
    acadp_df = privatizer.privatize(df)

    logger.info("Budget allocation:")
    for i, (block, eps) in enumerate(zip(blocks, [privatizer.allocations[j] for j in range(len(blocks))])):
        logger.info(f"  Block {i} ({len(block)} features): ε={eps:.4f} ({eps/EPSILON*100:.1f}%)")

    # ========== WORKFLOW 4: EVALUATION ==========
    logger.info("=" * 60)
    logger.info("WORKFLOW 4: COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    # Generate baselines
    logger.info("Generating baselines...")
    baselines = run_all_baselines(
        df, bounds, EPSILON,
        mechanism=MECHANISM,
        random_state=RANDOM_STATE,
        n_random_trials=N_TRIALS,
    )
    uniform_df = baselines['uniform']
    random_df = baselines['random_blocking']

    # Three-way comparison
    logger.info("Computing three-way comparison at ε=%.1f..." % EPSILON)
    comparison = compare_three_way(df, acadp_df, uniform_df, random_df, columns=NUMERIC_COLS)

    # Print improvement summary
    print("\n" + "=" * 70)
    print("  ACADP EVALUATION RESULTS (ε = %.1f)" % EPSILON)
    print("=" * 70)

    metric_labels = {
        'mean_mae': 'Mean Absolute Error',
        'mean_mse': 'Mean Squared Error',
        'correlation_error': 'Correlation Error',
        'mean_kl_divergence': 'KL Divergence',
        'mean_js_divergence': 'Jensen-Shannon Divergence',
        'mean_wasserstein_distance': 'Wasserstein Distance',
        'mean_ks_statistic': 'KS Statistic',
        'mean_query_error': 'Statistical Query Error',
    }

    print("\n  %-35s %10s %10s %10s %12s %12s" % ('Metric', 'ACADP', 'Uniform', 'Random', 'vs Uniform', 'vs Random'))
    print("  " + "-" * 93)

    for key, label in metric_labels.items():
        a = comparison['acadp_metrics'].get(key)
        u = comparison['uniform_metrics'].get(key)
        r = comparison['random_metrics'].get(key)
        if a is not None and u is not None:
            imp_u = (u - a) / u * 100 if u != 0 else 0
            imp_r = (r - a) / r * 100 if r and r != 0 else 0
            symbol_u = "+" if imp_u > 0 else ""
            symbol_r = "+" if imp_r > 0 else ""
            print("  %-35s %10.4f %10.4f %10.4f %11s%% %11s%%" % (
                label, a, u, r if r else 0,
                "%s%.1f" % (symbol_u, imp_u),
                "%s%.1f" % (symbol_r, imp_r)
            ))

    # Multi-epsilon sweep
    logger.info("Running multi-ε sweep: %s" % EPSILON_VALUES)
    sweep_results = run_multi_epsilon_sweep(
        df, blocks, bounds,
        epsilon_values=EPSILON_VALUES,
        mechanism=MECHANISM,
        allocation_method=ALLOCATION_METHOD,
        random_state=RANDOM_STATE,
        n_trials=N_TRIALS,
        columns=NUMERIC_COLS,
    )

    # Print sweep summary
    print("\n" + "=" * 70)
    print("  MULTI-ε SWEEP RESULTS")
    print("=" * 70)
    print("\n  %-8s %12s %12s %12s %15s" % ('ε', 'ACADP MAE', 'Uniform MAE', 'Random MAE', 'Improvement'))
    print("  " + "-" * 63)

    for eps in EPSILON_VALUES:
        r = sweep_results[eps]
        a_mae = r['acadp_metrics'].get('mean_mae', 0)
        u_mae = r['uniform_metrics'].get('mean_mae', 0)
        r_mae = r['random_metrics'].get('mean_mae', 0)
        imp = (u_mae - a_mae) / u_mae * 100 if u_mae else 0
        print("  ε=%-5.1f %12.4f %12.4f %12.4f %14.1f%%" % (eps, a_mae, u_mae, r_mae, imp))

    # Downstream ML evaluation
    logger.info("Running downstream ML evaluation...")
    privatized_dfs = {
        'ACADP': acadp_df,
        'Uniform': uniform_df,
        'Random_Blocking': random_df,
    }
    ml_results = evaluate_downstream_ml(
        df, privatized_dfs,
        regression_target='fare_amount',
        classification_target='payment_type',
        feature_cols=[c for c in NUMERIC_COLS if c not in ['fare_amount', 'payment_type']],
        random_state=RANDOM_STATE,
    )

    # Print ML results
    print("\n" + "=" * 70)
    print("  DOWNSTREAM ML PERFORMANCE")
    print("=" * 70)

    for method, results in ml_results.items():
        if 'regression' in results and results['regression']:
            r = results['regression']
            print("  %s - Regression (fare_amount): RMSE=%.4f, R²=%.4f" % (
                method, r.get('privatized_rmse', 0), r.get('privatized_r2', 0)))
        if 'classification' in results and results['classification']:
            c = results['classification']
            print("  %s - Classification (payment_type): Acc=%.4f, F1=%.4f" % (
                method, c.get('privatized_accuracy', 0), c.get('privatized_f1', 0)))

    # Generate all plots
    logger.info("Generating publication-quality plots...")
    import os
    os.makedirs('output/plots', exist_ok=True)

    generate_all_plots(
        original_df=df,
        acadp_df=acadp_df,
        uniform_df=uniform_df,
        random_df=random_df,
        comparison=comparison,
        blocks=blocks,
        allocations=privatizer.allocations,
        total_epsilon=EPSILON,
        columns=NUMERIC_COLS,
        sweep_results=sweep_results,
        ml_results=ml_results,
        output_dir='output/plots',
    )

    # Save all results
    save_comparison(comparison, 'output/comparison.json')
    save_comparison(sweep_results, 'output/sweep_results.json')
    save_comparison(ml_results, 'output/ml_results.json')

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print("  Total time: %.1f seconds" % total_elapsed)
    print("  Results saved to: output/")
    print("  Plots saved to: output/plots/")
    print("=" * 70)


if __name__ == '__main__':
    main()
