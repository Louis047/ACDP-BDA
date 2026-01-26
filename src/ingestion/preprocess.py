"""
Encoding and normalization of features.
Prepares data for correlation analysis and DP mechanisms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles encoding and normalization of features."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.feature_info: Dict[str, Dict] = {}
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with categorical columns
            columns: List of columns to encode (None = auto-detect)
            method: 'label' for label encoding, 'onehot' for one-hot
        
        Returns:
            DataFrame with encoded columns
        """
        processed_df = df.copy()
        
        if columns is None:
            # Auto-detect categorical columns
            columns = [
                col for col in df.columns
                if df[col].dtype == 'object' or df[col].dtype.name == 'category'
            ]
        
        if not columns:
            logger.info("No categorical columns to encode")
            return processed_df
        
        logger.info(f"Encoding {len(columns)} categorical columns using {method} encoding")
        
        if method == 'label':
            for col in columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    processed_df[col] = self.label_encoders[col].fit_transform(
                        processed_df[col].astype(str)
                    )
                else:
                    processed_df[col] = self.label_encoders[col].transform(
                        processed_df[col].astype(str)
                    )
                
                # Store encoding info
                self.feature_info[col] = {
                    'type': 'categorical',
                    'encoding': 'label',
                    'n_categories': len(self.label_encoders[col].classes_)
                }
        
        elif method == 'onehot':
            # One-hot encoding
            for col in columns:
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df = processed_df.drop(columns=[col])
                logger.info(f"One-hot encoded '{col}' into {len(dummies.columns)} columns")
        
        return processed_df
    
    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Normalize numeric features.
        
        Args:
            df: DataFrame with numeric columns
            columns: List of columns to normalize (None = all numeric)
            method: 'standard' (z-score) or 'minmax' (0-1 scaling)
        
        Returns:
            DataFrame with normalized columns
        """
        processed_df = df.copy()
        
        if columns is None:
            # Select only numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.info("No numeric columns to normalize")
            return processed_df
        
        logger.info(f"Normalizing {len(columns)} columns using {method} scaling")
        
        for col in columns:
            if col not in processed_df.columns:
                continue
            
            if method == 'standard':
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    processed_df[col] = self.scalers[col].fit_transform(
                        processed_df[[col]]
                    ).flatten()
                else:
                    processed_df[col] = self.scalers[col].transform(
                        processed_df[[col]]
                    ).flatten()
            
            elif method == 'minmax':
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                    processed_df[col] = self.scalers[col].fit_transform(
                        processed_df[[col]]
                    ).flatten()
                else:
                    processed_df[col] = self.scalers[col].transform(
                        processed_df[[col]]
                    ).flatten()
            
            # Store normalization info
            if col not in self.feature_info:
                self.feature_info[col] = {}
            self.feature_info[col]['normalization'] = method
        
        return processed_df
    
    def handle_missing(
        self,
        df: pd.DataFrame,
        strategy: str = 'drop',
        fill_value: Optional[Union[float, str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'drop' (drop rows), 'fill' (fill with value), 'mean' (fill with mean)
            fill_value: Value to use if strategy='fill'
        
        Returns:
            DataFrame with handled missing values
        """
        processed_df = df.copy()
        
        missing_count = processed_df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values detected")
            return processed_df
        
        logger.info(f"Handling {missing_count} missing values using strategy: {strategy}")
        
        if strategy == 'drop':
            processed_df = processed_df.dropna()
            logger.info(f"Dropped rows with missing values. New shape: {processed_df.shape}")
        
        elif strategy == 'fill':
            if fill_value is None:
                raise ValueError("fill_value required when strategy='fill'")
            processed_df = processed_df.fillna(fill_value)
        
        elif strategy == 'mean':
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
                processed_df[numeric_cols].mean()
            )
            # For categorical, use mode
            categorical_cols = processed_df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                mode_value = processed_df[col].mode()
                if len(mode_value) > 0:
                    processed_df[col] = processed_df[col].fillna(mode_value[0])
        
        return processed_df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        encode_categorical: bool = True,
        normalize: bool = False,
        handle_missing: bool = True
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            encode_categorical: Whether to encode categorical features
            normalize: Whether to normalize numeric features
            handle_missing: Whether to handle missing values
        
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        if handle_missing:
            processed_df = self.handle_missing(processed_df)
        
        if encode_categorical:
            processed_df = self.encode_categorical(processed_df)
        
        if normalize:
            processed_df = self.normalize(processed_df)
        
        logger.info(f"Preprocessing complete. Final shape: {processed_df.shape}")
        return processed_df


def preprocess_dataframe(
    df: pd.DataFrame,
    encode_categorical: bool = True,
    normalize: bool = False,
    handle_missing: bool = True
) -> pd.DataFrame:
    """
    Convenience function for preprocessing.
    
    Args:
        df: Raw DataFrame
        encode_categorical: Whether to encode categorical features
        normalize: Whether to normalize numeric features
        handle_missing: Whether to handle missing values
    
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(
        df,
        encode_categorical=encode_categorical,
        normalize=normalize,
        handle_missing=handle_missing
    )
