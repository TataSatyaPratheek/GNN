# processmine/data/features.py (NEW FILE)

import numpy as np
import torch
import time


def normalize_features(features, method='l2', in_place=False, return_tensor=False):
    """
    Normalize features with different methods optimized for memory efficiency.
    
    Args:
        features: NumPy array or Pandas DataFrame of features to normalize
        method: Normalization method ('l2', 'standard', 'minmax', 'none')
        in_place: Whether to modify the input array in-place when possible
        return_tensor: Whether to return a PyTorch tensor instead of NumPy array
        
    Returns:
        Normalized features as NumPy array or PyTorch tensor
    """
    # Convert to numpy array if needed
    if hasattr(features, 'values'):
        features = features.values
    
    # Early return if no normalization
    if method is None or method.lower() == 'none':
        return torch.tensor(features) if return_tensor else features
    
    # Make a copy if not in_place
    if not in_place and features.flags.writeable:
        features = features.copy()
    
    if method.lower() == 'l2':
        # L2 normalization - compute norms first
        norms = np.sqrt(np.sum(features**2, axis=1, keepdims=True))
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        
        # Normalize
        if in_place and features.flags.writeable:
            np.divide(features, norms, out=features)
            normalized = features
        else:
            normalized = features / norms
            
    elif method.lower() == 'standard':
        # Standard scaling
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        # Avoid division by zero
        std = np.maximum(std, 1e-10)
        
        # Normalize
        if in_place and features.flags.writeable:
            np.subtract(features, mean, out=features)
            np.divide(features, std, out=features)
            normalized = features
        else:
            normalized = (features - mean) / std
            
    elif method.lower() == 'minmax':
        # Min-max scaling to [0, 1]
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        range_vals = np.maximum(max_vals - min_vals, 1e-10)
        
        # Normalize
        if in_place and features.flags.writeable:
            np.subtract(features, min_vals, out=features)
            np.divide(features, range_vals, out=features)
            normalized = features
        else:
            normalized = (features - min_vals) / range_vals
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Convert to PyTorch tensor if requested
    if return_tensor:
        normalized = torch.tensor(normalized, dtype=torch.float32)
    
    return normalized

def extract_features(df, feature_cols=None, categorical_cols=None, time_features=True):
    """
    Extract and preprocess features from a process dataframe.
    
    Args:
        df: Pandas DataFrame with process data
        feature_cols: List of columns to use as features (default: None for auto-detection)
        categorical_cols: List of categorical columns to one-hot encode (default: None)
        time_features: Whether to extract time-based features (default: True)
        
    Returns:
        DataFrame with processed features
    """
    import pandas as pd
    
    # Make a copy to avoid modifying original
    result_df = df.copy()
    
    # Auto-detect feature columns if not provided
    if feature_cols is None:
        feature_cols = [col for col in df.columns if not col.startswith('case_') 
                        and col not in ['timestamp', 'next_task', 'next_timestamp']]
    
    # Extract time features if requested
    if time_features and 'timestamp' in df.columns:
        # Add time-based features
        result_df['hour_of_day'] = df['timestamp'].dt.hour
        result_df['day_of_week'] = df['timestamp'].dt.dayofweek
        result_df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        result_df['month'] = df['timestamp'].dt.month
        
        # Add these to feature columns
        time_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'month']
        feature_cols = feature_cols + time_cols
    
    # Handle categorical columns with one-hot encoding
    if categorical_cols is not None:
        for col in categorical_cols:
            if col in df.columns:
                # Get dummies and add prefix
                dummies = pd.get_dummies(df[col], prefix=col)
                # Add to result
                result_df = pd.concat([result_df, dummies], axis=1)
                # Update feature columns
                feature_cols = [c for c in feature_cols if c != col] + list(dummies.columns)
    
    return result_df, feature_cols