"""
Basic dataset statistics.
Provides summary information for validation and debugging.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistics for a DataFrame.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dict containing various statistics
    """
    stats = {
        'shape': df.shape,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numeric statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
        stats['numeric_ranges'] = {
            col: {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
            for col in numeric_cols
        }
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        stats['categorical_stats'] = {
            col: {
                'n_unique': int(df[col].nunique()),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'freq': int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0
            }
            for col in categorical_cols
        }
    
    return stats


def print_stats(df: pd.DataFrame) -> None:
    """
    Print formatted statistics to console.
    
    Args:
        df: DataFrame to analyze
    """
    stats = compute_basic_stats(df)
    
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Shape: {stats['shape']}")
    print(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB")
    print(f"\nColumns ({stats['n_columns']}):")
    for col in stats['columns']:
        print(f"  - {col} ({stats['dtypes'][col]})")
    
    if 'numeric_stats' in stats:
        print(f"\nNumeric Columns ({len(stats['numeric_ranges'])}):")
        for col, ranges in stats['numeric_ranges'].items():
            print(f"  {col}:")
            print(f"    Range: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
            print(f"    Mean: {ranges['mean']:.4f}, Std: {ranges['std']:.4f}")
    
    if 'categorical_stats' in stats:
        print(f"\nCategorical Columns ({len(stats['categorical_stats'])}):")
        for col, cat_stats in stats['categorical_stats'].items():
            print(f"  {col}:")
            print(f"    Unique values: {cat_stats['n_unique']}")
            if cat_stats['most_frequent'] is not None:
                print(f"    Most frequent: {cat_stats['most_frequent']} (count: {cat_stats['freq']})")
    
    missing = {k: v for k, v in stats['missing_values'].items() if v > 0}
    if missing:
        print(f"\nMissing Values:")
        for col, count in missing.items():
            pct = stats['missing_percentage'][col]
            print(f"  {col}: {count} ({pct:.2f}%)")
    
    print("=" * 60)


def save_stats(stats: Dict[str, Any], filepath: str) -> None:
    """
    Save statistics to JSON file.
    
    Args:
        stats: Statistics dict
        filepath: Output file path
    """
    import json
    from pathlib import Path
    
    # Convert numpy types to native Python types for JSON
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif pd.api.types.is_datetime64_any_dtype(type(obj)):
            return str(obj)
        else:
            return obj
    
    stats_serializable = convert_types(stats)
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    logger.info(f"Statistics saved to {filepath}")
