"""
Enforce feature bounds (DP requirement).
All features must have known bounds for sensitivity calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class BoundsEnforcer:
    """Enforces and tracks bounds for all features (required for DP)."""
    
    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize with optional known bounds.
        
        Args:
            bounds: Dict mapping column names to (min, max) tuples
        """
        self.bounds: Dict[str, Tuple[float, float]] = bounds or {}
    
    def estimate_bounds(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        margin: float = 0.1
    ) -> Dict[str, Tuple[float, float]]:
        """
        Estimate bounds from data (with optional margin).
        
        Args:
            df: DataFrame to analyze
            columns: Columns to estimate (None = all numeric)
            margin: Fractional margin to add (e.g., 0.1 = 10% margin)
        
        Returns:
            Dict mapping column names to (min, max) tuples
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        estimated_bounds = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Add margin
            range_val = col_max - col_min
            margin_val = range_val * margin if range_val > 0 else abs(col_max) * margin if col_max != 0 else 1.0
            
            estimated_bounds[col] = (
                col_min - margin_val,
                col_max + margin_val
            )
        
        logger.info(f"Estimated bounds for {len(estimated_bounds)} columns")
        return estimated_bounds
    
    def enforce_bounds(
        self,
        df: pd.DataFrame,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        clip: bool = True
    ) -> pd.DataFrame:
        """
        Enforce bounds on DataFrame (clip values if necessary).
        
        Args:
            df: DataFrame to enforce bounds on
            bounds: Optional bounds dict (uses self.bounds if None)
            clip: If True, clip values to bounds; if False, raise error on violation
        
        Returns:
            DataFrame with enforced bounds
        """
        if bounds is None:
            bounds = self.bounds
        
        if not bounds:
            logger.warning("No bounds provided. Estimating from data...")
            bounds = self.estimate_bounds(df)
            self.bounds = bounds
        
        enforced_df = df.copy()
        violations = []
        
        for col, (min_val, max_val) in bounds.items():
            if col not in enforced_df.columns:
                continue
            
            if clip:
                # Clip values to bounds
                before_min = (enforced_df[col] < min_val).sum()
                before_max = (enforced_df[col] > max_val).sum()
                
                enforced_df[col] = enforced_df[col].clip(lower=min_val, upper=max_val)
                
                if before_min > 0 or before_max > 0:
                    logger.warning(
                        f"Column '{col}': clipped {before_min} values below min, "
                        f"{before_max} values above max"
                    )
            else:
                # Check for violations
                below_min = (enforced_df[col] < min_val).sum()
                above_max = (enforced_df[col] > max_val).sum()
                
                if below_min > 0 or above_max > 0:
                    violations.append({
                        'column': col,
                        'below_min': below_min,
                        'above_max': above_max,
                        'bounds': (min_val, max_val)
                    })
        
        if violations and not clip:
            error_msg = "Bounds violations detected:\n"
            for v in violations:
                error_msg += (
                    f"  {v['column']}: {v['below_min']} below min, "
                    f"{v['above_max']} above max (bounds: {v['bounds']})\n"
                )
            raise ValueError(error_msg)
        
        logger.info(f"Enforced bounds for {len(bounds)} columns")
        return enforced_df
    
    def get_sensitivity(self, column: str) -> float:
        """
        Get sensitivity (range) for a column.
        
        Args:
            column: Column name
        
        Returns:
            Sensitivity (max - min)
        """
        if column not in self.bounds:
            raise ValueError(f"No bounds defined for column: {column}")
        
        min_val, max_val = self.bounds[column]
        return max_val - min_val
    
    def get_all_sensitivities(self) -> Dict[str, float]:
        """
        Get sensitivities for all columns.
        
        Returns:
            Dict mapping column names to sensitivities
        """
        return {
            col: self.get_sensitivity(col)
            for col in self.bounds.keys()
        }
    
    def save_bounds(self, filepath: str) -> None:
        """Save bounds to JSON file."""
        import json
        from pathlib import Path
        
        # Convert tuples to lists for JSON serialization
        bounds_dict = {
            col: [float(min_val), float(max_val)]
            for col, (min_val, max_val) in self.bounds.items()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(bounds_dict, f, indent=2)
        logger.info(f"Bounds saved to {filepath}")
    
    @classmethod
    def load_bounds(cls, filepath: str) -> 'BoundsEnforcer':
        """Load bounds from JSON file."""
        import json
        
        with open(filepath, 'r') as f:
            bounds_dict = json.load(f)
        
        # Convert lists back to tuples
        bounds = {
            col: (float(min_val), float(max_val))
            for col, [min_val, max_val] in bounds_dict.items()
        }
        
        return cls(bounds)


def enforce_feature_bounds(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    estimate_if_missing: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Convenience function to enforce bounds.
    
    Args:
        df: DataFrame to process
        bounds: Optional known bounds
        estimate_if_missing: If True, estimate bounds if not provided
    
    Returns:
        Tuple of (bounded DataFrame, bounds dict)
    """
    enforcer = BoundsEnforcer(bounds)
    
    if not enforcer.bounds and estimate_if_missing:
        enforcer.bounds = enforcer.estimate_bounds(df)
    
    bounded_df = enforcer.enforce_bounds(df)
    return bounded_df, enforcer.bounds
