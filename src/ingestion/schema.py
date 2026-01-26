"""
Schema validation and type enforcement.
Ensures data types are consistent and valid for DP processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates and enforces schema for DP-ready datasets."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with optional schema definition.
        
        Args:
            schema: Dict mapping column names to expected types
                   Types can be: 'int', 'float', 'str', 'category', etc.
        """
        self.schema = schema or {}
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Validated DataFrame with enforced types
        """
        logger.info(f"Validating schema for {len(df.columns)} columns")
        
        validated_df = df.copy()
        
        for col, expected_type in self.schema.items():
            if col not in validated_df.columns:
                logger.warning(f"Column '{col}' in schema but not in DataFrame")
                continue
            
            if expected_type == 'int':
                validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce').astype('Int64')
            elif expected_type == 'float':
                validated_df[col] = pd.to_numeric(validated_df[col], errors='coerce').astype('float64')
            elif expected_type == 'str':
                validated_df[col] = validated_df[col].astype('string')
            elif expected_type == 'category':
                validated_df[col] = validated_df[col].astype('category')
            else:
                logger.warning(f"Unknown type '{expected_type}' for column '{col}'")
        
        # Check for missing values
        missing = validated_df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        
        logger.info("Schema validation complete")
        return validated_df
    
    def infer_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Infer schema from DataFrame.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dict mapping column names to inferred types
        """
        schema = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_integer_dtype(dtype):
                schema[col] = 'int'
            elif pd.api.types.is_float_dtype(dtype):
                schema[col] = 'float'
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                # Check if it's actually categorical
                if df[col].nunique() / len(df) < 0.1:  # Less than 10% unique values
                    schema[col] = 'category'
                else:
                    schema[col] = 'str'
            elif pd.api.types.is_categorical_dtype(dtype):
                schema[col] = 'category'
            else:
                schema[col] = str(dtype)
        
        logger.info(f"Inferred schema for {len(schema)} columns")
        return schema
    
    def save_schema(self, filepath: str) -> None:
        """Save schema to JSON file."""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.schema, f, indent=2)
        logger.info(f"Schema saved to {filepath}")
    
    @classmethod
    def load_schema(cls, filepath: str) -> 'SchemaValidator':
        """Load schema from JSON file."""
        import json
        
        with open(filepath, 'r') as f:
            schema = json.load(f)
        return cls(schema)


def validate_dataframe(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Convenience function for schema validation.
    
    Args:
        df: DataFrame to validate
        schema: Optional schema dict
    
    Returns:
        Validated DataFrame
    """
    validator = SchemaValidator(schema)
    return validator.validate(df)
