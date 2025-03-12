"""
Optimized data loading and preprocessing for process mining with intelligent chunking 
and memory management.
"""
import pandas as pd
import numpy as np
import torch
import time
import logging
import gc
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any, Optional, List, Union
from functools import partial

logger = logging.getLogger(__name__)

def load_and_preprocess_data(
    data_path: str, 
    norm_method: str = 'l2',
    chunk_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_dtypes: bool = True,
    memory_limit_gb: float = 4.0
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Load and preprocess event log data with optimized memory usage
    
    Args:
        data_path: Path to the data file (CSV)
        norm_method: Normalization method ('l2', 'standard', 'minmax', or None)
        chunk_size: Size of chunks to process (auto-detected if None)
        cache_dir: Directory to cache intermediate results (None for no caching)
        use_dtypes: Whether to optimize dtypes to reduce memory usage
        memory_limit_gb: Memory limit in GB for chunking calculation
        
    Returns:
        Tuple of (preprocessed_df, task_encoder, resource_encoder)
    """
    logger.info(f"Loading data from {data_path}")
    start_time = time.time()
    
    # Check if cached preprocessing exists
    cached_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        file_hash = hash(os.path.abspath(data_path) + str(os.path.getmtime(data_path)))
        cached_file = os.path.join(cache_dir, f"processed_{file_hash}.pkl")
        
        if os.path.exists(cached_file):
            try:
                logger.info(f"Loading cached preprocessed data from {cached_file}")
                df, task_encoder, resource_encoder = pd.read_pickle(cached_file)
                logger.info(f"Loaded cached data: {len(df):,} rows, {df['case_id'].nunique():,} cases")
                return df, task_encoder, resource_encoder
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
    
    # Determine optimal chunk size based on file size and available memory
    if chunk_size is None:
        file_size_gb = os.path.getsize(data_path) / (1024 ** 3)
        # Estimate memory needed (CSV parsing typically expands by 2-4x)
        estimated_memory_gb = file_size_gb * 3
        
        if estimated_memory_gb > memory_limit_gb:
            # Need to use chunking
            rows_per_gb = 1_000_000  # Approximate rows per GB, adjust based on column count
            chunk_size = int(memory_limit_gb * rows_per_gb / estimated_memory_gb)
            # Round to a nice number
            chunk_size = max(10000, round(chunk_size, -4))
            logger.info(f"File size: {file_size_gb:.2f} GB, using chunk size: {chunk_size:,} rows")
        else:
            # Can load the whole file
            chunk_size = None
            logger.info(f"File size: {file_size_gb:.2f} GB, loading in single pass")
    
    # Optimize dtypes if requested
    dtype_optimizers = None
    if use_dtypes:
        # Sample the first few rows to infer optimal dtypes
        sample = pd.read_csv(data_path, nrows=1000)
        dtype_optimizers = _optimize_dtypes(sample)
        del sample
    
    # Load data with efficient chunking for large files
    if chunk_size:
        # Process in chunks
        chunks = []
        total_rows = 0
        
        # Use iterator for large files
        for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size, dtype=dtype_optimizers)):
            # Apply initial column standardization to handle various formats
            chunk = _standardize_columns(chunk)
            
            # Track progress
            total_rows += len(chunk)
            logger.info(f"Loaded chunk {i+1}: {len(chunk):,} rows, total: {total_rows:,} rows")
            
            chunks.append(chunk)
            
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    else:
        # Load entire file at once
        df = pd.read_csv(data_path, dtype=dtype_optimizers)
        df = _standardize_columns(df)
    
    logger.info(f"Data loaded: {len(df):,} rows")
    
    # Preprocess data
    df, task_encoder, resource_encoder = _preprocess_data(df, norm_method)
    
    # Cache result if requested
    if cached_file:
        try:
            df.to_pickle(cached_file)
            logger.info(f"Cached preprocessed data to {cached_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    # Log data statistics
    preprocessing_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {preprocessing_time:.2f}s")
    logger.info(f"Data statistics:")
    logger.info(f"  Cases: {df['case_id'].nunique():,}")
    logger.info(f"  Activities: {len(task_encoder.classes_):,}")
    logger.info(f"  Resources: {len(resource_encoder.classes_):,}")
    logger.info(f"  Events: {len(df):,}")
    logger.info(f"  Time range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"  Memory usage: {memory_usage:.2f} MB")
    
    return df, task_encoder, resource_encoder

def _optimize_dtypes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create optimized dtypes for loading CSV files efficiently
    
    Args:
        df: Sample dataframe to infer dtypes from
        
    Returns:
        Dictionary of column->dtype mappings for pd.read_csv
    """
    dtype_map = {}
    
    for col in df.columns:
        # Skip timestamp columns
        if col.lower().endswith('timestamp') or col.lower().endswith('time'):
            continue
            
        # Check if column is categorical or ID
        if col.lower().endswith('id') or df[col].nunique() < len(df) * 0.1:
            sample_val = df[col].iloc[0]
            
            if pd.api.types.is_integer_dtype(df[col]):
                # Use smallest possible int dtype
                max_val = df[col].max()
                min_val = df[col].min()
                
                if min_val >= 0:
                    if max_val < 2**8:
                        dtype_map[col] = 'uint8'
                    elif max_val < 2**16:
                        dtype_map[col] = 'uint16'
                    elif max_val < 2**32:
                        dtype_map[col] = 'uint32'
                else:
                    if min_val > -2**7 and max_val < 2**7:
                        dtype_map[col] = 'int8'
                    elif min_val > -2**15 and max_val < 2**15:
                        dtype_map[col] = 'int16'
                    elif min_val > -2**31 and max_val < 2**31:
                        dtype_map[col] = 'int32'
            
            elif pd.api.types.is_string_dtype(df[col]):
                # Use category for string columns with few unique values
                if df[col].nunique() < 1000:
                    dtype_map[col] = 'category'
        
        # For float columns
        elif pd.api.types.is_float_dtype(df[col]):
            # Check if float32 is sufficient
            if df[col].min() > -3.4e38 and df[col].max() < 3.4e38:
                dtype_map[col] = 'float32'
    
    return dtype_map

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to handle different naming conventions (e.g., XES format)
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with standardized column names
    """
    # Handle different column naming conventions (support XES format)
    column_mappings = {
        "case:concept:name": "case_id",
        "concept:name": "task_name",
        "org:resource": "resource",
        "time:timestamp": "timestamp",
        "case:id": "case_id"
    }
    
    df = df.rename(columns=column_mappings, errors="ignore")
    
    # Check for required columns
    required_columns = ["case_id", "task_name", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df

def _preprocess_data(df: pd.DataFrame, norm_method: str = 'l2') -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Core preprocessing function for process mining data
    
    Args:
        df: Input dataframe
        norm_method: Normalization method ('l2', 'standard', 'minmax', or None)
        
    Returns:
        Tuple of (preprocessed_df, task_encoder, resource_encoder)
    """
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Remove rows with invalid timestamps
    n_before = len(df)
    df = df.dropna(subset=["timestamp"])
    n_after = len(df)
    
    if n_before > n_after:
        logger.warning(f"Removed {n_before - n_after:,} rows with invalid timestamps")
    
    # Sort by case_id and timestamp
    df = df.sort_values(["case_id", "timestamp"])
    
    # Encode categorical variables efficiently
    task_encoder = LabelEncoder()
    resource_encoder = LabelEncoder()
    
    # Apply encoding in place to save memory
    df["task_id"] = task_encoder.fit_transform(df["task_name"])
    df["resource_id"] = resource_encoder.fit_transform(df["resource"])
    
    # Add derived temporal features efficiently
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Add next_task by case
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    
    # Remove rows with no next task (last event in each case)
    n_before = len(df)
    df = df.dropna(subset=["next_task"])
    n_after = len(df)
    
    logger.info(f"Removed {n_before - n_after:,} end events (no next task)")
    
    # Convert next_task to int
    df["next_task"] = df["next_task"].astype(int)
    
    # Create feature columns
    feature_cols = ["task_id", "resource_id", "day_of_week", "hour_of_day", "is_weekend"]
    
    # Add amount feature if available
    if "amount" in df.columns:
        feature_cols.append("amount")
    
    # Get feature matrix
    features = df[feature_cols].values
    
    # Normalize features
    if norm_method and norm_method.lower() != 'none':
        norm_features = _normalize_features(features, method=norm_method)
        
        # Add normalized features to dataframe
        for i, col in enumerate(feature_cols):
            df[f"feat_{col}"] = norm_features[:, i]
    else:
        # Still add the feature columns with original values for consistency
        for i, col in enumerate(feature_cols):
            df[f"feat_{col}"] = df[col]
    
    # Add case-level features
    df = _add_case_features(df)
    
    return df, task_encoder, resource_encoder

def _normalize_features(features: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize features with different methods
    
    Args:
        features: Feature array to normalize
        method: Normalization method ('l2', 'standard', 'minmax')
        
    Returns:
        Normalized feature array
    """
    if method.lower() == 'l2':
        # L2 normalization
        norms = np.sqrt((features ** 2).sum(axis=1, keepdims=True))
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return features / norms
    
    elif method.lower() == 'standard':
        # Standard scaling (zero mean, unit variance)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-10)  # Avoid division by zero
        return (features - mean) / std
    
    elif method.lower() == 'minmax':
        # Min-max scaling to [0, 1]
        min_vals = features.min(axis=0, keepdims=True)
        max_vals = features.max(axis=0, keepdims=True)
        denom = np.maximum(max_vals - min_vals, 1e-10)  # Avoid division by zero
        return (features - min_vals) / denom
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _add_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add case-level features for better prediction
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with additional case-level features
    """
    # Calculate case-level statistics
    case_stats = df.groupby('case_id').agg({
        'timestamp': ['min', 'max', 'count'],
        'task_id': 'nunique',
        'resource_id': 'nunique'
    })
    
    # Flatten multi-level columns
    case_stats.columns = ['_'.join(col).strip() for col in case_stats.columns.values]
    
    # Calculate duration
    case_stats['case_duration_seconds'] = (
        case_stats['timestamp_max'] - case_stats['timestamp_min']
    ).dt.total_seconds()
    
    # Rename columns for clarity
    case_stats = case_stats.rename(columns={
        'timestamp_count': 'case_events',
        'task_id_nunique': 'case_unique_tasks',
        'resource_id_nunique': 'case_unique_resources'
    })
    
    # Select columns to add back
    case_features = case_stats[[
        'case_events', 
        'case_unique_tasks', 
        'case_unique_resources', 
        'case_duration_seconds'
    ]]
    
    # Add back to original dataframe
    return df.join(case_features, on='case_id')

def create_data_loader(
    graphs: List, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2
):
    """
    Create an optimized data loader for graph data
    
    Args:
        graphs: List of graph data objects
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        PyTorch Geometric DataLoader
    """
    from torch_geometric.loader import DataLoader
    
    # Auto-tune num_workers if not specified
    if num_workers == 0 and torch.cuda.is_available():
        import multiprocessing
        num_workers = min(4, multiprocessing.cpu_count())
    
    # Auto-enable pin_memory for GPU
    if torch.cuda.is_available() and not pin_memory:
        pin_memory = True
    
    # Create data loader
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

def create_sequence_dataset(df, max_seq_len=20, min_seq_len=2, feature_cols=None):
    """
    Create sequence dataset for LSTM models with optimized memory usage
    
    Args:
        df: Preprocessed dataframe
        max_seq_len: Maximum sequence length
        min_seq_len: Minimum sequence length
        feature_cols: Feature columns to use (default: all feat_ columns)
        
    Returns:
        Tuple of (sequences, targets, seq_lengths)
    """
    logger.info("Creating sequence dataset")
    start_time = time.time()
    
    # If feature columns not specified, use all feat_ columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    # Group by case and create sequences
    sequences = []
    targets = []
    seq_lengths = []
    
    # Create sequences based on case_id
    for case_id, group in df.groupby('case_id'):
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Skip if sequence is too short
        if len(group) < min_seq_len:
            continue
        
        # Create sequence
        seq = group[feature_cols].values
        # Truncate if too long
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
        
        # Get target (next task after sequence)
        target = group['next_task'].values
        if len(target) > max_seq_len:
            target = target[:max_seq_len]
        
        # Store sequence, target, and length
        sequences.append(torch.FloatTensor(seq))
        targets.append(torch.LongTensor(target))
        seq_lengths.append(len(seq))
    
    logger.info(f"Created {len(sequences)} sequences in {time.time() - start_time:.2f}s")
    
    return sequences, targets, seq_lengths