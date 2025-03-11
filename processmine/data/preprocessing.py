# data/preprocessing.py - More efficient preprocessor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

def create_feature_representation(df: pd.DataFrame, use_norm_features: bool = True):
    """
    Create features with optimized, vectorized operations
    
    Args:
        df: Process data dataframe
        use_norm_features: Whether to use L2 normalization
        
    Returns:
        Tuple of (transformed_df, task_encoder, resource_encoder)
    """
    # Avoid unnecessary copying
    # Process data in place when possible
    
    # Extract temporal features efficiently
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Encode categorical features
    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"])

    # Compute next tasks efficiently
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    
    # Handle missing values from last events in cases
    missing_next = df["next_task"].isna()
    if missing_next.any():
        print(f"Dropping {missing_next.sum()} rows with no next task (end events)")
        df = df[~missing_next].copy()
    
    df["next_task"] = df["next_task"].astype(int)

    # Feature columns
    feature_cols = ["task_id", "resource_id", "day_of_week", "hour_of_day"]
    
    # Add amount if available
    if "amount" in df.columns:
        feature_cols.append("amount")
    
    # Get feature values
    features = df[feature_cols].values
    
    # Apply normalization in one operation 
    if use_norm_features:
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

    # Add features to dataframe
    for i, col in enumerate(feature_cols):
        df[f"feat_{col}"] = normalized_features[:, i]

    return df, le_task, le_resource