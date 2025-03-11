#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data preprocessing module for process mining
Handles data loading, cleaning, feature engineering and XES format
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, RobustScaler
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import time
import gc
import logging
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path, use_adaptive_norm=False, enhanced_features=False,
                           enhanced_graphs=False, batch_size=None, required_cols=None):
    """
    Load and preprocess the event log data with progress feedback and XES format support
    
    Args:
        data_path: Path to data file
        use_adaptive_norm: Whether to use adaptive normalization
        enhanced_features: Whether to use enhanced feature engineering 
        enhanced_graphs: Whether to use enhanced graph building
        batch_size: Optional batch size for memory efficiency
        required_cols: List of required columns
        
    Returns:
        Tuple of (df, graphs, task_encoder, resource_encoder)
    """
    print("\n==== Loading and Preprocessing Data ====")
    start_time = time.time()
    
    if required_cols is None:
        required_cols = ["case_id", "task_name", "timestamp", "resource"]
    
    print(f"Loading data from {data_path}...")
    try:
        # Use more efficient CSV reading with chunking for large files
        chunksize = 100000
        chunks = []
        
        # Count total lines for progress bar
        with open(data_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # Read in chunks with progress bar
        with tqdm(total=total_lines, desc="Reading CSV", ncols=100) as pbar:
            for chunk in pd.read_csv(data_path, chunksize=chunksize):
                chunks.append(chunk)
                pbar.update(len(chunk))
                
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"\033[91mError reading CSV in chunks: {e}\033[0m")
        print("Falling back to standard pandas read_csv...")
        df = pd.read_csv(data_path)
    
    print(f"Data loaded: {len(df)} rows, {df.shape[1]} columns")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Rename columns for consistency (supporting XES format)
    df.rename(columns={
        "case:id": "case_id",
        "case:concept:name": "case_name",
        "concept:name": "task_name",
        "org:resource": "resource",
        "time:timestamp": "timestamp",
        "case:Amount": "amount",
        "case:BudgetNumber": "budget_number",
        "case:DeclarationNumber": "declaration_number",
        "org:role": "role"
    }, inplace=True, errors="ignore")

    print(f"Columns after renaming: {df.columns.tolist()}")

    # Validate required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}. Found columns: {df.columns.tolist()}")

    # Process timestamps
    print("Processing timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    before_na = len(df)
    df.dropna(subset=["timestamp"], inplace=True)
    after_na = len(df)
    if before_na > after_na:
        print(f"\033[93mWarning: Dropped {before_na - after_na} rows with invalid timestamps\033[0m")
    
    # Sort by case ID and timestamp
    print("Sorting data...")
    df.sort_values(["case_id","timestamp"], inplace=True)
    
    # Report basic statistics
    case_count = df["case_id"].nunique()
    task_count = df["task_name"].nunique() 
    resource_count = df["resource"].nunique()
    
    # Calculate time period
    min_date = df["timestamp"].min().strftime('%Y-%m-%d')
    max_date = df["timestamp"].max().strftime('%Y-%m-%d')
    
    print(f"\033[1mData Summary\033[0m:")
    print(f"  Time period: \033[96m{min_date}\033[0m to \033[96m{max_date}\033[0m")
    print(f"  Cases: \033[96m{case_count:,}\033[0m")
    print(f"  Activities: \033[96m{task_count}\033[0m")
    print(f"  Resources: \033[96m{resource_count}\033[0m")
    print(f"  Events: \033[96m{len(df):,}\033[0m")
    
    # Get task and resource encoders
    task_encoder = LabelEncoder().fit(df["task_name"])
    resource_encoder = LabelEncoder().fit(df["resource"])
    
    # Create feature representation
    df, _, _ = create_feature_representation(df, use_norm_features=use_adaptive_norm)
    
    # Build graph data
    graphs = build_graph_data(df)
    
    print(f"Preprocessing completed in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return df, graphs, task_encoder, resource_encoder


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
    Create scaled or normalized feature representation with enhanced progress tracking
    
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
    
    # Time features with progress feedback
    print("Extracting temporal features...")
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Encode tasks and resources
    print("Encoding categorical features...")
    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"])

    # Add next task with progress feedback
    print("Computing next tasks...")
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    before_na = len(df)
    df.dropna(subset=["next_task"], inplace=True)
    after_na = len(df)
    
    if before_na > after_na:
        print(f"Dropped {before_na - after_na} rows with no next task (end events)")
    
    df["next_task"] = df["next_task"].astype(int)

    # Feature scaling with progress
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


def build_graph_data(df):
    """
    Convert preprocessed data into graph format for GNN with progress tracking
    
    Args:
        df: Process data dataframe
        
    Returns:
        List of graph data objects
    """
    print("\n==== Building Graph Data ====")
    start_time = time.time()
    
    graphs = []
    case_groups = df.groupby("case_id")
    num_cases = len(case_groups)
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure feature extraction was performed correctly.")
    
    # Create progress bar
    progress_bar = tqdm(
        case_groups, 
        desc="Building graphs",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100,
        total=num_cases
    )
    
    # Track statistics
    edge_counts = []
    node_counts = []
    
    for cid, cdata in progress_bar:
        cdata = cdata.sort_values("timestamp")

        # Create node features
        x_data = torch.tensor(cdata[feature_cols].values, dtype=torch.float)

        # Create edges between sequential activities (more efficient)
        n_nodes = len(cdata)
        node_counts.append(n_nodes)
        
        if n_nodes > 1:
            src = list(range(n_nodes-1))
            tgt = list(range(1,n_nodes))
            edge_index = torch.tensor([src+tgt, tgt+src], dtype=torch.long)
            edge_counts.append(2 * (n_nodes - 1))  # Bidirectional edges
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_counts.append(0)
            
        y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        data_obj = Data(x=x_data, edge_index=edge_index, y=y_data)
        graphs.append(data_obj)
        
        # Periodically collect garbage to prevent memory buildup
        if len(graphs) % 1000 == 0:
            gc.collect()

    # Report statistics
    avg_nodes = np.mean(node_counts)
    avg_edges = np.mean(edge_counts)
    max_nodes = np.max(node_counts)
    
    print(f"\033[1mGraph Statistics\033[0m:")
    print(f"  Total graphs: \033[96m{len(graphs):,}\033[0m")
    print(f"  Avg nodes per graph: \033[96m{avg_nodes:.2f}\033[0m")
    print(f"  Avg edges per graph: \033[96m{avg_edges:.2f}\033[0m")
    print(f"  Max nodes in a graph: \033[96m{max_nodes}\033[0m")
    print(f"Graphs built in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return graphs


def compute_class_weights(df, num_classes):
    """
    Compute balanced class weights for training with improved efficiency
    
    Args:
        df: Preprocessed dataframe
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\nComputing class weights...")
    start_time = time.time()
    
    # Extract labels more efficiently
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
    
    # Keep weights on CPU initially - will be moved to device in setup_optimizer_and_loss if needed
    return torch.tensor(class_weights, dtype=torch.float32)