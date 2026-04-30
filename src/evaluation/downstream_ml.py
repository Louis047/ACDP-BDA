"""
Downstream ML utility evaluation.
Measures how well privatized data preserves ML model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error as sklearn_mse,
    r2_score, mean_absolute_error as sklearn_mae
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

# Try to import xgboost, fallback to sklearn
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available, using RandomForest only")


def evaluate_classification(
    original_df: pd.DataFrame,
    privatized_df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate classification accuracy on original vs privatized data.

    Train on privatized, test on original (holdout from original).
    This measures how well a model trained on privatized data generalizes.

    Args:
        original_df: Original dataset
        privatized_df: Privatized dataset
        target_col: Target column for classification
        feature_cols: Feature columns (None = all numeric except target)
        test_size: Test split ratio
        random_state: Random seed

    Returns:
        Dict with accuracy and F1 scores
    """
    if target_col not in original_df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return {}

    if feature_cols is None:
        feature_cols = [
            col for col in original_df.select_dtypes(include=[np.number]).columns
            if col != target_col and col in privatized_df.columns
        ]

    if not feature_cols:
        logger.warning("No feature columns for classification")
        return {}

    # Prepare data
    X_orig = original_df[feature_cols].values
    y_orig = original_df[target_col].values
    X_priv = privatized_df[feature_cols].values
    y_priv = privatized_df[target_col].values

    # Handle NaN
    mask_orig = ~np.any(np.isnan(X_orig), axis=1) & ~np.isnan(y_orig)
    mask_priv = ~np.any(np.isnan(X_priv), axis=1) & ~np.isnan(y_priv)
    X_orig, y_orig = X_orig[mask_orig], y_orig[mask_orig]
    X_priv, y_priv = X_priv[mask_priv], y_priv[mask_priv]

    if len(X_orig) < 100 or len(X_priv) < 100:
        logger.warning("Not enough data for classification evaluation")
        return {}

    # Split original data for testing
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=test_size, random_state=random_state
    )

    # Also split privatized data (same size for fair comparison)
    n_train = len(X_train_orig)
    X_train_priv = X_priv[:n_train]
    y_train_priv = y_priv[:n_train]

    results = {}

    # Train on original → test
    clf_orig = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = clf_orig.predict(X_test)
    results['original_accuracy'] = float(accuracy_score(y_test, y_pred_orig))
    results['original_f1'] = float(f1_score(y_test, y_pred_orig, average='weighted', zero_division=0))

    # Train on privatized → test on SAME test set from original
    clf_priv = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf_priv.fit(X_train_priv, y_train_priv)
    y_pred_priv = clf_priv.predict(X_test)
    results['privatized_accuracy'] = float(accuracy_score(y_test, y_pred_priv))
    results['privatized_f1'] = float(f1_score(y_test, y_pred_priv, average='weighted', zero_division=0))

    # Accuracy drop
    results['accuracy_drop'] = results['original_accuracy'] - results['privatized_accuracy']
    results['f1_drop'] = results['original_f1'] - results['privatized_f1']

    logger.info(
        f"Classification: orig_acc={results['original_accuracy']:.4f}, "
        f"priv_acc={results['privatized_accuracy']:.4f}, "
        f"drop={results['accuracy_drop']:.4f}"
    )

    return results


def evaluate_regression(
    original_df: pd.DataFrame,
    privatized_df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate regression performance on original vs privatized data.

    Args:
        original_df: Original dataset
        privatized_df: Privatized dataset
        target_col: Target column for regression
        feature_cols: Feature columns
        test_size: Test split ratio
        random_state: Random seed

    Returns:
        Dict with RMSE, MAE, and R² scores
    """
    if target_col not in original_df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return {}

    if feature_cols is None:
        feature_cols = [
            col for col in original_df.select_dtypes(include=[np.number]).columns
            if col != target_col and col in privatized_df.columns
        ]

    if not feature_cols:
        logger.warning("No feature columns for regression")
        return {}

    # Prepare data
    X_orig = original_df[feature_cols].values
    y_orig = original_df[target_col].values
    X_priv = privatized_df[feature_cols].values
    y_priv = privatized_df[target_col].values

    # Handle NaN
    mask_orig = ~np.any(np.isnan(X_orig), axis=1) & ~np.isnan(y_orig)
    mask_priv = ~np.any(np.isnan(X_priv), axis=1) & ~np.isnan(y_priv)
    X_orig, y_orig = X_orig[mask_orig], y_orig[mask_orig]
    X_priv, y_priv = X_priv[mask_priv], y_priv[mask_priv]

    if len(X_orig) < 100 or len(X_priv) < 100:
        logger.warning("Not enough data for regression evaluation")
        return {}

    # Split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=test_size, random_state=random_state
    )

    n_train = len(X_train_orig)
    X_train_priv = X_priv[:n_train]
    y_train_priv = y_priv[:n_train]

    results = {}

    # Train on original
    reg_orig = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    reg_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = reg_orig.predict(X_test)
    results['original_rmse'] = float(np.sqrt(sklearn_mse(y_test, y_pred_orig)))
    results['original_mae'] = float(sklearn_mae(y_test, y_pred_orig))
    results['original_r2'] = float(r2_score(y_test, y_pred_orig))

    # Train on privatized
    reg_priv = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    reg_priv.fit(X_train_priv, y_train_priv)
    y_pred_priv = reg_priv.predict(X_test)
    results['privatized_rmse'] = float(np.sqrt(sklearn_mse(y_test, y_pred_priv)))
    results['privatized_mae'] = float(sklearn_mae(y_test, y_pred_priv))
    results['privatized_r2'] = float(r2_score(y_test, y_pred_priv))

    # Performance drops
    results['rmse_increase'] = results['privatized_rmse'] - results['original_rmse']
    results['r2_drop'] = results['original_r2'] - results['privatized_r2']

    logger.info(
        f"Regression: orig_rmse={results['original_rmse']:.4f}, "
        f"priv_rmse={results['privatized_rmse']:.4f}, "
        f"orig_r2={results['original_r2']:.4f}, priv_r2={results['privatized_r2']:.4f}"
    )

    return results


def evaluate_downstream_ml(
    original_df: pd.DataFrame,
    privatized_dfs: Dict[str, pd.DataFrame],
    classification_target: Optional[str] = None,
    regression_target: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Evaluate downstream ML performance across all privatization methods.

    Args:
        original_df: Original dataset
        privatized_dfs: Dict mapping method name to privatized DataFrame
        classification_target: Column for classification task
        regression_target: Column for regression task
        feature_cols: Feature columns
        random_state: Random seed

    Returns:
        Dict with ML performance metrics per method
    """
    results = {}

    for method_name, priv_df in privatized_dfs.items():
        method_results = {}

        if classification_target:
            logger.info(f"Evaluating classification for {method_name}...")
            clf_results = evaluate_classification(
                original_df, priv_df, classification_target,
                feature_cols=feature_cols, random_state=random_state
            )
            method_results['classification'] = clf_results

        if regression_target:
            logger.info(f"Evaluating regression for {method_name}...")
            reg_results = evaluate_regression(
                original_df, priv_df, regression_target,
                feature_cols=feature_cols, random_state=random_state
            )
            method_results['regression'] = reg_results

        results[method_name] = method_results

    return results
