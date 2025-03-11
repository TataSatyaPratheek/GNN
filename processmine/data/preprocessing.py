#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing utilities for process mining
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, RobustScaler
import torch
import time
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class AdaptiveNormalizer:
    """
    Adaptive normalization based on data characteristics
    Selects appropriate normalization method based on data statistics
    """
    def __init__(self, strategies=None, auto_detect=True):
        """
        Initialize adaptive normalizer
        
        Args:
            strategies: Dictionary of normalization strategies
            auto_detect: Whether to automatically detect the best strategy
        """
        # Default normalization strategies
        if strategies is None:
            self.strategies = {
                'robust': RobustScaler(),
                'l2': Normalizer(norm='l2'),
                'minmax': MinMaxScaler(feature_range=(0, 1))
            }
        else:
            self.strategies = strategies
        
        self.auto_detect = auto_detect
        self.selected_strategy = None
        self.strategy_name = None
        self.feature_statistics = {}
        self.is_fitted = False
    
    def fit(self, features, feature_names=None):
        """
        Fit normalizer to data
        
        Args:
            features: Feature array [num_samples, num_features]
            feature_names: Optional feature names
            
        Returns:
            self
        """
        # Calculate feature statistics
        self._compute_feature_statistics(features, feature_names)
        
        # Select normalization strategy
        if self.auto_detect:
            self.strategy_name = self._select_best_strategy()
            self.selected_strategy = self.strategies[self.strategy_name]
        else:
            # Default to MinMax
            self.strategy_name = 'minmax'
            self.selected_strategy = self.strategies['minmax']
        
        # Fit the selected strategy
        self.selected_strategy.fit(features)
        self.is_fitted = True
        
        print(f"Selected normalization strategy: {self.strategy_name}")
        
        return self
    
    def transform(self, features):
        """
        Transform features using the selected strategy
        
        Args:
            features: Feature array [num_samples, num_features]
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fit before transform")
        
        # Handle NaN values
        if np.isnan(features).any():
            print(f"Found {np.isnan(features).sum()} NaN values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0)
        
        # Handle infinite values
        if np.isinf(features).any():
            print(f"Found {np.isinf(features).sum()} infinite values. Replacing with large values.")
            features = np.nan_to_num(features, posinf=1e6, neginf=-1e6)
        
        return self.selected_strategy.transform(features)
    
    def fit_transform(self, features, feature_names=None):
        """
        Fit and transform in one step
        
        Args:
            features: Feature array [num_samples, num_features]
            feature_names: Optional feature names
            
        Returns:
            Transformed features
        """
        return self.fit(features, feature_names).transform(features)
    
    def _compute_feature_statistics(self, features, feature_names=None):
        """
        Compute feature statistics for strategy selection
        
        Args:
            features: Feature array
            feature_names: Optional feature names
        """
        # Handle NaN and infinite values
        features_clean = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Calculate statistics
        self.feature_statistics = {
            'mean': np.mean(features_clean, axis=0),
            'std': np.std(features_clean, axis=0),
            'min': np.min(features_clean, axis=0),
            'max': np.max(features_clean, axis=0),
            'skewness': stats.skew(features_clean, axis=0),
            'kurtosis': stats.kurtosis(features_clean, axis=0),
            'range_ratio': np.max(features_clean, axis=0) / (np.min(features_clean, axis=0) + 1e-8)
        }
        
        # Create detailed stats by feature
        self.feature_details = {}
        for i, name in enumerate(feature_names):
            self.feature_details[name] = {
                'mean': self.feature_statistics['mean'][i],
                'std': self.feature_statistics['std'][i],
                'min': self.feature_statistics['min'][i],
                'max': self.feature_statistics['max'][i],
                'skewness': self.feature_statistics['skewness'][i],
                'kurtosis': self.feature_statistics['kurtosis'][i],
                'range_ratio': self.feature_statistics['range_ratio'][i]
            }
    
    def _select_best_strategy(self):
        """
        Select the best normalization strategy based on feature statistics
        
        Returns:
            Strategy name
        """
        # Get statistics
        skewness = self.feature_statistics['skewness']
        range_ratio = self.feature_statistics['range_ratio']
        
        # Check for highly skewed data or large range differences
        if np.any(np.abs(skewness) > 1.5) or np.any(range_ratio > 10):
            # Highly skewed with large range differences - use robust scaling
            return 'robust'
        elif np.any(np.abs(self.feature_statistics['mean']) > 5.0):
            # Large magnitudes - use L2 normalization
            return 'l2'
        else:
            # Well-behaved features - use MinMax
            return 'minmax'

def create_feature_representation(df, use_norm_features=True):
    """
    Create scaled or normalized feature representation
    
    Args:
        df: Process data dataframe
        use_norm_features: Whether to use L2 normalization
        
    Returns:
        Tuple of (transformed_df, task_encoder, resource_encoder)
    """
    print("\n==== Creating Feature Representation ====")
    start_time = time.time()
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Time features
    print("Extracting temporal features...")
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Encode tasks and resources
    print("Encoding categorical features...")
    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"])

    # Add next task
    print("Computing next tasks...")
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    before_na = len(df)
    df.dropna(subset=["next_task"], inplace=True)
    after_na = len(df)
    
    if before_na > after_na:
        print(f"Dropped {before_na - after_na} rows with no next task (end events)")
    
    df["next_task"] = df["next_task"].astype(int)

    # Feature scaling
    print("Scaling features...")
    feature_cols = ["task_id", "resource_id"]
    
    # Add amount if available
    if "amount" in df.columns:
        feature_cols.append("amount")
    
    feature_cols.extend(["day_of_week", "hour_of_day"])
    
    raw_features = df[feature_cols].values

    # Diagnose data issues
    nans = np.isnan(raw_features).sum()
    if nans > 0:
        print(f"\033[93mWarning: Found {nans} NaN values in features. Replacing with zeros.\033[0m")
        raw_features = np.nan_to_num(raw_features, nan=0.0)
    
    infs = np.isinf(raw_features).sum()
    if infs > 0:
        print(f"\033[93mWarning: Found {infs} infinite values in features. Replacing with large values.\033[0m")
        raw_features = np.nan_to_num(raw_features, posinf=1e6, neginf=-1e6)
    
    # Analyze feature characteristics
    feature_means = np.mean(raw_features, axis=0)
    feature_stds = np.std(raw_features, axis=0)
    feature_mins = np.min(raw_features, axis=0)
    feature_maxs = np.max(raw_features, axis=0)
    
    print("Feature statistics:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: mean={feature_means[i]:.4f}, std={feature_stds[i]:.4f}, min={feature_mins[i]:.4f}, max={feature_maxs[i]:.4f}")
    
    # Adaptive normalization based on data characteristics
    if use_norm_features:
        print("Using L2 normalization...")
        normalizer = Normalizer(norm='l2')
        combined_features = normalizer.fit_transform(raw_features)
    else:
        # Check if we should use robust scaling
        if np.any(feature_maxs / np.maximum(feature_mins, 1e-10) > 100):
            print("Data has extreme values - using RobustScaler...")
            scaler = RobustScaler()
        else:
            print("Using MinMaxScaler...")
            scaler = MinMaxScaler()
        combined_features = scaler.fit_transform(raw_features)

    # Add features back to dataframe
    for i, col in enumerate(feature_cols):
        df[f"feat_{col}"] = combined_features[:, i]

    print(f"Feature representation created in \033[96m{time.time() - start_time:.2f}s\033[0m")
    return df, le_task, le_resource

def compute_class_weights(df, num_classes):
    """
    Compute balanced class weights for training with imbalanced data
    
    Args:
        df: Preprocessed dataframe
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\nComputing class weights...")
    start_time = time.time()
    
    # Extract labels
    train_labels = df["next_task"].values
    
    # Count class frequencies
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    # Report class distribution
    total = len(train_labels)
    print(f"Class distribution ({len(unique_labels)} classes):")
    for label, count in zip(unique_labels[:5], counts[:5]):
        print(f"  Class {label}: {count:,} samples ({count/total*100:.2f}%)")
    if len(unique_labels) > 5:
        print(f"  ... and {len(unique_labels)-5} more classes")
    
    # Compute weights
    class_weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = compute_class_weight("balanced", classes=present, y=train_labels)
    
    for i, cval in enumerate(present):
        class_weights[cval] = cw[i]
    
    # Report weight range
    min_weight = np.min(class_weights[class_weights > 0])
    max_weight = np.max(class_weights)
    print(f"Class weight range: {min_weight:.4f} - {max_weight:.4f}")
    print(f"Class weights computed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    # Keep weights on CPU initially
    return torch.tensor(class_weights, dtype=torch.float32)