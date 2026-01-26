"""
Batch loading of large datasets (CSV/Parquet).
Handles large files efficiently using chunked reading.
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


def load_csv(
    filepath: Union[str, Path],
    chunksize: Optional[int] = None,
    **kwargs
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Load CSV file, optionally in chunks for large files.
    
    Args:
        filepath: Path to CSV file
        chunksize: If provided, return iterator over chunks of this size
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame or iterator of DataFrames
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    logger.info(f"Loading CSV file: {filepath}")
    
    if chunksize:
        logger.info(f"Using chunked reading with chunksize={chunksize}")
        return pd.read_csv(filepath, chunksize=chunksize, **kwargs)
    else:
        return pd.read_csv(filepath, **kwargs)


def load_parquet(
    filepath: Union[str, Path],
    columns: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load Parquet file efficiently.
    
    Args:
        filepath: Path to Parquet file
        columns: Optional list of columns to read
        **kwargs: Additional arguments passed to pd.read_parquet
    
    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    
    logger.info(f"Loading Parquet file: {filepath}")
    
    if columns:
        logger.info(f"Reading {len(columns)} columns")
    
    return pd.read_parquet(filepath, columns=columns, **kwargs)


def save_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    **kwargs
) -> None:
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        **kwargs: Additional arguments passed to df.to_parquet
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving DataFrame to Parquet: {filepath}")
    logger.info(f"Shape: {df.shape}")
    
    df.to_parquet(filepath, **kwargs)
    logger.info("Save complete")


def load_data(
    filepath: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Auto-detect file type and load accordingly.
    
    Args:
        filepath: Path to data file
        file_type: Optional override ('csv' or 'parquet')
        **kwargs: Additional arguments for loading
    
    Returns:
        DataFrame or iterator
    """
    filepath = Path(filepath)
    
    if file_type is None:
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            file_type = 'csv'
        elif suffix in ['.parquet', '.pqt']:
            file_type = 'parquet'
        else:
            raise ValueError(f"Unknown file type: {suffix}. Specify file_type explicitly.")
    
    if file_type == 'csv':
        return load_csv(filepath, **kwargs)
    elif file_type == 'parquet':
        return load_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load dataset")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", help="Output Parquet path (optional)")
    parser.add_argument("--file-type", choices=['csv', 'parquet'], help="File type override")
    parser.add_argument("--chunksize", type=int, help="Chunk size for CSV reading")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    df = load_data(args.input, file_type=args.file_type, chunksize=args.chunksize)
    
    if args.output:
        if isinstance(df, Iterator):
            # For chunked reading, concatenate first
            df = pd.concat(list(df), ignore_index=True)
        save_parquet(df, args.output)
    else:
        print(f"Loaded dataset: {df.shape}")
