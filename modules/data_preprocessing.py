#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced data preprocessing module for process mining
Implements adaptive normalization and optimized feature engineering
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, RobustScaler
from scipy import stats
import time
import gc
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
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
        
        logger.info(f"Selected normalization strategy: {self.strategy_name}")
        
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
            logger.warning(f"Found {np.isnan(features).sum()} NaN values. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0)
        
        # Handle infinite values
        if np.isinf(features).any():
            logger.warning(f"Found {np.isinf(features).sum()} infinite values. Replacing with large values.")
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


class EnhancedFeatureEngineering:
    """
    Enhanced feature engineering for process mining
    Extracts and transforms features from process data
    """
    def __init__(self, temporal_features=True, attribute_features=True,
                 graph_features=True, normalize_features=True):
        """
        Initialize feature engineering
        
        Args:
            temporal_features: Whether to include temporal features
            attribute_features: Whether to include attribute features
            graph_features: Whether to include graph-based features
            normalize_features: Whether to normalize features
        """
        self.temporal_features = temporal_features
        self.attribute_features = attribute_features
        self.graph_features = graph_features
        self.normalize_features = normalize_features
        
        # Feature encoders
        self.task_encoder = None
        self.resource_encoder = None
        
        # Feature normalizers
        self.feature_normalizer = AdaptiveNormalizer() if normalize_features else None
        
        # Feature statistics
        self.feature_stats = {}
        self.is_fitted = False
    
    def fit(self, df):
        """
        Fit feature engineering to data
        
        Args:
            df: Process data dataframe
            
        Returns:
            self
        """
        logger.info("Fitting enhanced feature engineering")
        start_time = time.time()
        
        # Encode categorical features
        logger.info("Encoding categorical features")
        self.task_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        
        self.task_encoder.fit(df['task_name'])
        self.resource_encoder.fit(df['resource'])
        
        # Extract and fit to sample features
        sample_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df
        sample_features = self._extract_features(sample_df, fit=True)
        
        # Fit normalizer if enabled
        if self.normalize_features and self.feature_normalizer is not None:
            logger.info("Fitting feature normalizer")
            self.feature_normalizer.fit(sample_features)
        
        self.is_fitted = True
        logger.info(f"Feature engineering fitted in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def transform(self, df):
        """
        Transform process data with enhanced features
        
        Args:
            df: Process data dataframe
            
        Returns:
            Transformed dataframe with enhanced features
        """
        if not self.is_fitted:
            raise ValueError("Feature engineering must be fit before transform")
        
        logger.info("Transforming data with enhanced features")
        start_time = time.time()
        
        # Copy dataframe to avoid modifying original
        df_transformed = df.copy()
        
        # Encode tasks and resources
        df_transformed['task_id'] = self.task_encoder.transform(df_transformed['task_name'])
        df_transformed['resource_id'] = self.resource_encoder.transform(df_transformed['resource'])
        
        # Extract features
        features = self._extract_features(df_transformed, fit=False)
        
        # Normalize features if enabled
        if self.normalize_features and self.feature_normalizer is not None:
            features = self.feature_normalizer.transform(features)
        
        # Add feature columns to dataframe
        feature_names = self._get_feature_names()
        for i, name in enumerate(feature_names):
            df_transformed[f'feat_{name}'] = features[:, i]
        
        logger.info(f"Data transformed in {time.time() - start_time:.2f} seconds")
        
        return df_transformed
    
    def fit_transform(self, df):
        """
        Fit and transform in one step
        
        Args:
            df: Process data dataframe
            
        Returns:
            Transformed dataframe with enhanced features
        """
        return self.fit(df).transform(df)
    
    def _extract_features(self, df, fit=False):
        """
        Extract features from process data
        
        Args:
            df: Process data dataframe
            fit: Whether this is being called during fit
            
        Returns:
            Feature array
        """
        features_list = []
        
        # Basic features: task ID and resource ID
        task_features = df['task_id'].values.reshape(-1, 1)
        resource_features = df['resource_id'].values.reshape(-1, 1)
        
        features_list.append(task_features)
        features_list.append(resource_features)
        
        # Amount features if available
        if 'amount' in df.columns:
            amount_features = df['amount'].values.reshape(-1, 1)
            features_list.append(amount_features)
        
        # Temporal features
        if self.temporal_features and 'timestamp' in df.columns:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract temporal features
            hour_of_day = df['timestamp'].dt.hour.values.reshape(-1, 1)
            day_of_week = df['timestamp'].dt.dayofweek.values.reshape(-1, 1)
            
            # Extract time since case start
            time_features = []
            for case_id, case_group in df.groupby('case_id'):
                case_group = case_group.sort_values('timestamp')
                case_start = case_group['timestamp'].min()
                time_since_start = (case_group['timestamp'] - case_start).dt.total_seconds() / 3600  # hours
                time_features.append(time_since_start.values)
            
            time_since_start = np.concatenate(time_features).reshape(-1, 1)
            
            features_list.append(hour_of_day)
            features_list.append(day_of_week)
            features_list.append(time_since_start)
        
        # Attribute features
        if self.attribute_features:
            # Case complexity: number of activities in case so far
            complexity_features = []
            for case_id, case_group in df.groupby('case_id'):
                case_group = case_group.sort_values('timestamp')
                activity_count = np.arange(1, len(case_group) + 1)
                complexity_features.append(activity_count)
            
            case_complexity = np.concatenate(complexity_features).reshape(-1, 1)
            features_list.append(case_complexity)
            
            # Case resource diversity: number of unique resources in case so far
            diversity_features = []
            for case_id, case_group in df.groupby('case_id'):
                case_group = case_group.sort_values('timestamp')
                resource_diversity = np.array([
                    len(set(case_group['resource_id'].values[:i+1]))
                    for i in range(len(case_group))
                ])
                diversity_features.append(resource_diversity)
            
            resource_diversity = np.concatenate(diversity_features).reshape(-1, 1)
            features_list.append(resource_diversity)
        
        # Graph features - simple version
        if self.graph_features:
            # Create transition count matrix
            task_transition_matrix = np.zeros((
                len(self.task_encoder.classes_),
                len(self.task_encoder.classes_)
            ))
            
            # Fill transition matrix
            for case_id, case_group in df.groupby('case_id'):
                case_group = case_group.sort_values('timestamp')
                tasks = case_group['task_id'].values
                
                for i in range(len(tasks) - 1):
                    src, dst = tasks[i], tasks[i+1]
                    task_transition_matrix[src, dst] += 1
            
            # Normalize by row sums
            row_sums = task_transition_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_probs = task_transition_matrix / row_sums
            
            # Calculate node in-degree and out-degree centrality
            in_degree = task_transition_matrix.sum(axis=0)
            out_degree = task_transition_matrix.sum(axis=1)
            
            # Add centrality as features
            in_degree_features = np.array([in_degree[task] for task in df['task_id']]).reshape(-1, 1)
            out_degree_features = np.array([out_degree[task] for task in df['task_id']]).reshape(-1, 1)
            
            features_list.append(in_degree_features)
            features_list.append(out_degree_features)
        
        # Combine all features
        features = np.hstack(features_list)
        
        # Store feature statistics during fit
        if fit:
            self.feature_stats = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0),
                'min': np.min(features, axis=0),
                'max': np.max(features, axis=0)
            }
        
        return features
    
    def _get_feature_names(self):
        """
        Get feature names
        
        Returns:
            List of feature names
        """
        feature_names = ['task_id', 'resource_id', 'amount']
        
        if self.temporal_features:
            feature_names.extend(['hour', 'day_of_week', 'time_since_start'])
        
        if self.attribute_features:
            feature_names.extend(['case_complexity', 'resource_diversity'])
        
        if self.graph_features:
            feature_names.extend(['in_degree', 'out_degree'])
        
        return feature_names


class EnhancedGraphBuilder:
    """
    Enhanced graph builder for process mining
    Builds graph data with optimized memory usage
    """
    def __init__(self, add_self_loops=True, add_reverse_edges=True,
                 add_temporal_edges=False, max_temporal_distance=3,
                 add_edge_features=True, chunk_size=1000):
        """
        Initialize graph builder
        
        Args:
            add_self_loops: Whether to add self-loops
            add_reverse_edges: Whether to add reverse edges
            add_temporal_edges: Whether to add temporal edges based on time proximity
            max_temporal_distance: Maximum temporal distance for temporal edges
            add_edge_features: Whether to add edge features
            chunk_size: Chunk size for processing large datasets
        """
        self.add_self_loops = add_self_loops
        self.add_reverse_edges = add_reverse_edges
        self.add_temporal_edges = add_temporal_edges
        self.max_temporal_distance = max_temporal_distance
        self.add_edge_features = add_edge_features
        self.chunk_size = chunk_size
    
    def build_graphs(self, df, node_features, batch_size=None):
        """
        Build graph data from process data
        
        Args:
            df: Process data dataframe
            node_features: List of node feature columns
            batch_size: Optional batch size for memory efficiency
            
        Returns:
            List of graph data objects
        """
        logger.info("Building enhanced graph data")
        start_time = time.time()
        
        # Group by case
        case_ids = df['case_id'].unique()
        num_cases = len(case_ids)
        
        # Process in chunks for memory efficiency
        chunk_size = batch_size or self.chunk_size
        num_chunks = (num_cases + chunk_size - 1) // chunk_size
        
        all_graphs = []
        
        for chunk_idx in range(num_chunks):
            logger.info(f"Processing chunk {chunk_idx+1}/{num_chunks}")
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, num_cases)
            chunk_case_ids = case_ids[chunk_start:chunk_end]
            
            # Filter to chunk cases
            chunk_df = df[df['case_id'].isin(chunk_case_ids)]
            
            # Process each case in this chunk
            chunk_graphs = []
            for case_id in tqdm(chunk_case_ids, desc="Building graphs"):
                case_df = chunk_df[chunk_df['case_id'] == case_id].sort_values('timestamp')
                
                # Skip cases with only one activity
                if len(case_df) <= 1:
                    continue
                
                # Create graph
                graph = self._build_case_graph(case_df, node_features)
                chunk_graphs.append(graph)
            
            all_graphs.extend(chunk_graphs)
            
            # Force garbage collection between chunks
            gc.collect()
        
        logger.info(f"Built {len(all_graphs)} graphs in {time.time() - start_time:.2f} seconds")
        
        return all_graphs
    
    def _build_case_graph(self, case_df, node_features):
        """
        Build graph for a single case
        
        Args:
            case_df: Dataframe for a single case
            node_features: List of node feature columns
            
        Returns:
            PyG Data object
        """
        # Extract node features
        x = torch.tensor(case_df[node_features].values, dtype=torch.float)
        
        # Extract node indices for edge creation
        node_indices = np.arange(len(case_df))
        
        # Create edges (sequential by default)
        edges = []
        edge_attr = []
        
        # Sequential edges
        for i in range(len(node_indices) - 1):
            # Forward edge
            edges.append((node_indices[i], node_indices[i+1]))
            
            # Add time difference as edge attribute if enabled
            if self.add_edge_features and 'timestamp' in case_df.columns:
                time_diff = (case_df.iloc[i+1]['timestamp'] - case_df.iloc[i]['timestamp']).total_seconds() / 3600  # hours
                edge_attr.append([time_diff])
            
            # Reverse edge if enabled
            if self.add_reverse_edges:
                edges.append((node_indices[i+1], node_indices[i]))
                
                if self.add_edge_features and 'timestamp' in case_df.columns:
                    edge_attr.append([time_diff])  # Same time difference
        
        # Self-loops if enabled
        if self.add_self_loops:
            for i in range(len(node_indices)):
                edges.append((node_indices[i], node_indices[i]))
                
                if self.add_edge_features:
                    edge_attr.append([0.0])  # Zero time difference for self-loops
        
        # Temporal edges based on time proximity if enabled
        if self.add_temporal_edges and len(node_indices) > 2 and 'timestamp' in case_df.columns:
            timestamps = pd.to_datetime(case_df['timestamp']).values
            
            for i in range(len(node_indices)):
                for j in range(i+2, len(node_indices)):  # Skip direct neighbors
                    # Check temporal distance
                    temporal_distance = j - i
                    if temporal_distance <= self.max_temporal_distance:
                        # Add temporal edge
                        edges.append((node_indices[i], node_indices[j]))
                        
                        if self.add_edge_features:
                            time_diff = (timestamps[j] - timestamps[i]).total_seconds() / 3600
                            edge_attr.append([time_diff])
        
        # Convert edges to tensor format [2, num_edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Convert edge attributes if available
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else None
        
        # Extract target (next activity) for each node
        y = torch.zeros(len(case_df), dtype=torch.long)
        y[:-1] = torch.tensor(case_df['task_id'].values[1:], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_tensor,
            y=y
        )
        
        return data


# Combined preprocessing function
def load_and_preprocess_data(data_path, use_adaptive_norm=True, enhanced_features=True,
                           enhanced_graphs=True, batch_size=None):
    """
    Load and preprocess data with enhanced methods
    
    Args:
        data_path: Path to data file
        use_adaptive_norm: Whether to use adaptive normalization
        enhanced_features: Whether to use enhanced feature engineering
        enhanced_graphs: Whether to use enhanced graph building
        batch_size: Optional batch size for memory efficiency
        
    Returns:
        Tuple of (transformed_df, graphs, task_encoder, resource_encoder)
    """
    logger.info(f"Loading data from {data_path}")
    start_time = time.time()
    
    # Load data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    logger.info(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases "
               f"in {time.time() - start_time:.2f} seconds")
    
    # Rename columns for consistency if needed
    column_mapping = {
        "case:id": "case_id",
        "concept:name": "task_name",
        "time:timestamp": "timestamp",
        "org:resource": "resource",
        "case:Amount": "amount"
    }
    
    df = df.rename(columns={col: new_col for col, new_col in column_mapping.items() 
                           if col in df.columns and new_col not in df.columns})
    
    # Validate required columns
    required_cols = ["case_id", "task_name", "timestamp", "resource"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Process timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Check for invalid timestamps
        invalid_timestamps = df['timestamp'].isna().sum()
        if invalid_timestamps > 0:
            logger.warning(f"Found {invalid_timestamps} invalid timestamps. Removing rows.")
            df = df.dropna(subset=['timestamp'])
    
    # Sort by case ID and timestamp
    df = df.sort_values(['case_id', 'timestamp'])
    
    # Enhanced feature engineering
    if enhanced_features:
        feature_engineering = EnhancedFeatureEngineering(
            normalize_features=use_adaptive_norm
        )
        
        # Fit and transform
        df = feature_engineering.fit_transform(df)
        
        # Add next task feature
        df['next_task'] = df.groupby('case_id')['task_id'].shift(-1)
        df = df.dropna(subset=['next_task'])
        df['next_task'] = df['next_task'].astype(int)
        
    else:
        # Basic encoding
        task_encoder = LabelEncoder()
        resource_encoder = LabelEncoder()
        
        df['task_id'] = task_encoder.fit_transform(df['task_name'])
        df['resource_id'] = resource_encoder.fit_transform(df['resource'])
        
        # Add next task
        df['next_task'] = df.groupby('case_id')['task_id'].shift(-1)
        df = df.dropna(subset=['next_task'])
        df['next_task'] = df['next_task'].astype(int)
    
    # Get encoders
    task_encoder = LabelEncoder().fit(df['task_name'])
    resource_encoder = LabelEncoder().fit(df['resource'])
    
    # Build graphs
    if enhanced_graphs:
        graph_builder = EnhancedGraphBuilder(
            add_self_loops=True,
            add_reverse_edges=True,
            add_temporal_edges=True,
            add_edge_features=True
        )
        
        # Identify node feature columns
        node_feature_cols = [col for col in df.columns if col.startswith('feat_')]
        
        # Build graphs
        graphs = graph_builder.build_graphs(df, node_feature_cols, batch_size=batch_size)
    else:
        # Basic graph building
        graphs = []
        for case_id, case_data in df.groupby('case_id'):
            # Skip cases with only one event
            if len(case_data) <= 1:
                continue
            
            # Sort by timestamp
            case_data = case_data.sort_values('timestamp')
            
            # Get node features
            node_features = case_data[[col for col in case_data.columns 
                                     if col.startswith('feat_')]].values
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Create edges (sequential connections)
            edge_index = []
            for i in range(len(case_data) - 1):
                edge_index.append([i, i+1])
                if enhanced_graphs:  # Add reverse edges
                    edge_index.append([i+1, i])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            
            # Get target (next activity)
            y = torch.zeros(len(case_data), dtype=torch.long)
            y[:-1] = torch.tensor(case_data['task_id'].values[1:], dtype=torch.long)
            
            # Create graph
            data = Data(x=x, edge_index=edge_index, y=y)
            graphs.append(data)
    
    logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    return df, graphs, task_encoder, resource_encoder


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
    
    logger.info("Computing class weights")
    start_time = time.time()
    
    # Extract labels
    train_labels = df['next_task'].values
    
    # Count class frequencies
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    # Report class distribution
    total = len(train_labels)
    logger.info(f"Class distribution ({len(unique_labels)} classes):")
    
    for label, count in zip(unique_labels[:5], counts[:5]):
        logger.info(f"  Class {label}: {count:,} samples ({count/total*100:.2f}%)")
    
    if len(unique_labels) > 5:
        logger.info(f"  ... and {len(unique_labels)-5} more classes")
    
    # Compute weights using sklearn
    class_weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(train_labels)
    
    # Check for large class imbalance
    if max(counts) / min(counts) > 100:
        logger.warning("Extreme class imbalance detected. Using square root scaling for weights.")
        # Use square root scaling to avoid extreme weights
        sample_weights = np.sqrt(total / (len(unique_labels) * counts))
        for i, label in enumerate(unique_labels):
            class_weights[label] = sample_weights[i]
    else:
        # Use standard balanced weighting
        cw = compute_class_weight('balanced', classes=present, y=train_labels)
        for i, cval in enumerate(present):
            class_weights[cval] = cw[i]
    
    # Report weight range
    min_weight = np.min(class_weights[class_weights > 0])
    max_weight = np.max(class_weights)
    logger.info(f"Class weight range: {min_weight:.4f} - {max_weight:.4f}")
    logger.info(f"Class weights computed in {time.time() - start_time:.2f}s")
    
    return torch.tensor(class_weights, dtype=torch.float32)


# Examples and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Create synthetic test data
    def create_test_data(num_cases=100, avg_events=10, num_tasks=10, num_resources=5):
        """Create synthetic test data for process mining"""
        np.random.seed(42)
        
        # Generate cases
        events = []
        case_ids = []
        
        for case_id in range(num_cases):
            # Random number of events for this case
            num_events = max(2, int(np.random.normal(avg_events, 3)))
            
            # Generate timestamp for case start
            start_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
            
            # Generate events for this case
            for event_idx in range(num_events):
                # Random task and resource
                task_id = np.random.randint(0, num_tasks)
                resource_id = np.random.randint(0, num_resources)
                
                # Timestamp (sequential)
                timestamp = start_time + pd.Timedelta(hours=event_idx * 2 + np.random.randint(0, 2))
                
                # Random amount
                amount = np.random.uniform(10, 1000)
                
                # Add event
                events.append({
                    'case_id': f"case_{case_id}",
                    'task_name': f"task_{task_id}",
                    'resource': f"resource_{resource_id}",
                    'timestamp': timestamp,
                    'amount': amount
                })
                case_ids.append(f"case_{case_id}")
        
        # Create dataframe
        df = pd.DataFrame(events)
        
        return df
    
    # Create test data
    logger.info("Creating test data")
    df = create_test_data()
    
    # Test adaptive normalizer
    logger.info("\nTesting AdaptiveNormalizer...")
    
    # Create feature array
    features = np.vstack([
        df['amount'].values.reshape(-1, 1),
        np.random.normal(0, 1, size=(len(df), 1)),
        np.random.exponential(1, size=(len(df), 1))
    ]).T
    
    # Create normalizer and fit
    normalizer = AdaptiveNormalizer()
    normalized = normalizer.fit_transform(features)
    
    logger.info(f"Selected strategy: {normalizer.strategy_name}")
    logger.info(f"Feature shape: {features.shape}, Normalized shape: {normalized.shape}")
    
    # Test enhanced feature engineering
    logger.info("\nTesting EnhancedFeatureEngineering...")
    
    # Create and fit
    feature_engineering = EnhancedFeatureEngineering()
    df_transformed = feature_engineering.fit_transform(df)
    
    logger.info(f"Original columns: {df.columns.tolist()}")
    logger.info(f"Transformed columns: {[c for c in df_transformed.columns if c.startswith('feat_')]}")
    
    # Test enhanced graph builder
    logger.info("\nTesting EnhancedGraphBuilder...")
    
    # Create and build graphs
    graph_builder = EnhancedGraphBuilder()
    node_feature_cols = [c for c in df_transformed.columns if c.startswith('feat_')]
    graphs = graph_builder.build_graphs(df_transformed, node_feature_cols)
    
    logger.info(f"Built {len(graphs)} graphs")
    logger.info(f"First graph: {graphs[0]}")
    
    # Test combined preprocessing
    logger.info("\nTesting combined preprocessing...")
    
    # Save test data to csv
    temp_file = 'test_process_data.csv'
    df.to_csv(temp_file, index=False)
    
    # Load and preprocess
    df_processed, graphs, task_encoder, resource_encoder = load_and_preprocess_data(
        temp_file,
        use_adaptive_norm=True,
        enhanced_features=True,
        enhanced_graphs=True
    )
    
    logger.info(f"Processed dataframe shape: {df_processed.shape}")
    logger.info(f"Number of graphs: {len(graphs)}")
    logger.info(f"Task classes: {len(task_encoder.classes_)}")
    
    # Test class weights
    weights = compute_class_weights(df_processed, len(task_encoder.classes_))
    logger.info(f"Class weights shape: {weights.shape}")
    
    logger.info("All tests completed")
    
    # Clean up
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)