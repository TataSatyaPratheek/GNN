"""
Unified data loading and preprocessing for process mining
"""
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import logging
import time
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str, norm_features: bool = True) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Load and preprocess event log data
    
    Args:
        data_path: Path to the data file (CSV)
        norm_features: Whether to normalize features
        
    Returns:
        Tuple of (preprocessed_df, task_encoder, resource_encoder)
    """
    logger.info(f"Loading data from {data_path}")
    start_time = time.time()
    
    # Load data with efficient chunking for large files
    try:
        # For large files, use chunking
        chunks = []
        chunksize = 100000
        
        # Create progress bar
        chunk_iter = pd.read_csv(data_path, chunksize=chunksize)
        for chunk in tqdm(chunk_iter, desc="Loading data", unit="chunk"):
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
    except:
        # Fallback for smaller files
        df = pd.read_csv(data_path)
    
    logger.info(f"Data loaded: {len(df):,} rows")
    
    # Handle different column naming conventions (support XES format)
    column_mappings = {
        "case:concept:name": "case_id",
        "concept:name": "task_name",
        "org:resource": "resource",
        "time:timestamp": "timestamp",
        "case:id": "case_id"
    }
    
    df.rename(columns=column_mappings, inplace=True, errors="ignore")
    
    # Check for required columns
    required_columns = ["case_id", "task_name", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Remove rows with invalid timestamps
    n_before = len(df)
    df = df.dropna(subset=["timestamp"])
    n_after = len(df)
    
    if n_before > n_after:
        logger.warning(f"Removed {n_before - n_after} rows with invalid timestamps")
    
    # Sort by case_id and timestamp
    df = df.sort_values(["case_id", "timestamp"])
    
    # Encode categorical variables
    task_encoder = LabelEncoder()
    resource_encoder = LabelEncoder()
    
    df["task_id"] = task_encoder.fit_transform(df["task_name"])
    df["resource_id"] = resource_encoder.fit_transform(df["resource"])
    
    # Add derived features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour
    
    # Add next_task by case
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    
    # Remove rows with no next task (last event in each case)
    n_before = len(df)
    df = df.dropna(subset=["next_task"])
    n_after = len(df)
    
    logger.info(f"Removed {n_before - n_after} end events (no next task)")
    
    # Convert next_task to int
    df["next_task"] = df["next_task"].astype(int)
    
    # Create feature columns
    feature_cols = ["task_id", "resource_id", "day_of_week", "hour_of_day"]
    
    # Add amount feature if available
    if "amount" in df.columns:
        feature_cols.append("amount")
    
    # Get feature matrix
    features = df[feature_cols].values
    
    # Normalize features if requested
    if norm_features:
        # L2 normalization
        norms = np.sqrt((features ** 2).sum(axis=1, keepdims=True))
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized_features = features / norms
    else:
        # Standard scaling
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
        normalized_features = (features - mean) / std
    
    # Add normalized features to dataframe
    for i, col in enumerate(feature_cols):
        df[f"feat_{col}"] = normalized_features[:, i]
    
    # Log data statistics
    logger.info(f"Preprocessing completed in {time.time() - start_time:.2f}s")
    logger.info(f"Data statistics:")
    logger.info(f"  Cases: {df['case_id'].nunique():,}")
    logger.info(f"  Activities: {len(task_encoder.classes_):,}")
    logger.info(f"  Resources: {len(resource_encoder.classes_):,}")
    logger.info(f"  Events: {len(df):,}")
    logger.info(f"  Time range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    
    return df, task_encoder, resource_encoder