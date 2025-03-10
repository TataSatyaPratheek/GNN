#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline models for process mining
Includes decision trees, random forests, XGBoost, basic LSTM, and MLP models
for comparison against GNN implementations
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging
import gc


# Set up logging
logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Feature Engineering
#------------------------------------------------------------------------------

class ProcessFeatureExtractor:
    """
    Extract features from process data for ML models
    Creates tabular features from sequential process data
    """
    def __init__(self, max_seq_len=10, use_resources=True, use_time=True, 
                 use_attributes=True, categorical_encoding='onehot'):
        """
        Initialize feature extractor
        
        Args:
            max_seq_len: Maximum sequence length to consider
            use_resources: Whether to include resource information
            use_time: Whether to include time features
            use_attributes: Whether to include additional attributes
            categorical_encoding: How to encode categorical features ('onehot' or 'label')
        """
        self.max_seq_len = max_seq_len
        self.use_resources = use_resources
        self.use_time = use_time
        self.use_attributes = use_attributes
        self.categorical_encoding = categorical_encoding
        
        # To be populated during fit
        self.task_encoder = None
        self.resource_encoder = None
        self.num_tasks = 0
        self.num_resources = 0
        self.trained = False
    
    def fit(self, df):
        """
        Fit feature extractor to the data
        
        Args:
            df: Process data dataframe
            
        Returns:
            self
        """
        # Extract unique tasks and resources
        if self.categorical_encoding == 'onehot':
            from sklearn.preprocessing import OneHotEncoder
            
            # Fit task encoder
            self.task_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.task_encoder.fit(df[['task_id']])
            self.num_tasks = len(self.task_encoder.categories_[0])
            
            # Fit resource encoder if used
            if self.use_resources:
                self.resource_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                self.resource_encoder.fit(df[['resource_id']])
                self.num_resources = len(self.resource_encoder.categories_[0])
        
        elif self.categorical_encoding == 'label':
            # Already encoded as labels
            self.num_tasks = df['task_id'].nunique()
            if self.use_resources:
                self.num_resources = df['resource_id'].nunique()
        
        self.trained = True
        return self
    
    def transform(self, df):
        """
        Transform process data into features for ML models
        
        Args:
            df: Process data dataframe
            
        Returns:
            X: Features array
            y: Target array
        """
        if not self.trained:
            raise ValueError("Feature extractor must be fit before transform")
        
        # Group by case
        case_groups = df.groupby('case_id')
        
        # Lists to store features and targets
        X_list = []
        y_list = []
        
        # Process each case
        for case_id, case_data in case_groups:
            # Sort by timestamp
            case_data = case_data.sort_values('timestamp')
            
            # Skip cases with only one event (no next task)
            if len(case_data) <= 1:
                continue
            
            # Extract task sequence
            task_seq = case_data['task_id'].values
            
            # Process sub-sequences within the case
            for i in range(len(task_seq) - 1):
                # Extract prefix sequence (up to max_seq_len)
                prefix_idx = max(0, i + 1 - self.max_seq_len)
                prefix = task_seq[prefix_idx:i+1]
                
                # Get next task as target
                next_task = task_seq[i+1]
                
                # Create features for this sequence
                features = self._create_sequence_features(case_data.iloc[prefix_idx:i+1])
                
                # Store features and target
                X_list.append(features)
                y_list.append(next_task)
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _create_sequence_features(self, seq_df):
        """
        Create features from a sequence of events
        
        Args:
            seq_df: Dataframe for the event sequence
            
        Returns:
            features: Feature vector
        """
        features = []
        
        # Get task sequence
        task_seq = seq_df['task_id'].values
        
        # Pad/truncate to max_seq_len
        if len(task_seq) < self.max_seq_len:
            # Pad with -1 (will be encoded as zeros in one-hot)
            padded_tasks = np.pad(
                task_seq, 
                (self.max_seq_len - len(task_seq), 0),
                'constant', 
                constant_values=-1
            )
        else:
            padded_tasks = task_seq[-self.max_seq_len:]
        
        # Encode tasks
        if self.categorical_encoding == 'onehot':
            # Reshape for OneHotEncoder
            tasks_reshaped = padded_tasks.reshape(-1, 1)
            # Replace -1 with an out-of-range value
            tasks_reshaped[tasks_reshaped == -1] = self.num_tasks
            # One-hot encode
            encoded_tasks = []
            for task in tasks_reshaped:
                if task[0] == self.num_tasks:  # Padding
                    encoded_tasks.append(np.zeros(self.num_tasks))
                else:
                    encoded_tasks.append(
                        self.task_encoder.transform(task.reshape(1, -1))[0]
                    )
            encoded_tasks = np.vstack(encoded_tasks)
            # Flatten
            features.extend(encoded_tasks.flatten())
        else:
            # Use label encoding (normalized)
            normalized_tasks = padded_tasks.copy()
            normalized_tasks[normalized_tasks == -1] = 0  # Replace padding with 0
            normalized_tasks = normalized_tasks / self.num_tasks  # Normalize to [0,1]
            features.extend(normalized_tasks)
        
        # Add resource information if used
        if self.use_resources:
            resource_seq = seq_df['resource_id'].values
            
            # Pad/truncate to max_seq_len
            if len(resource_seq) < self.max_seq_len:
                padded_resources = np.pad(
                    resource_seq, 
                    (self.max_seq_len - len(resource_seq), 0),
                    'constant', 
                    constant_values=-1
                )
            else:
                padded_resources = resource_seq[-self.max_seq_len:]
            
            # Encode resources
            if self.categorical_encoding == 'onehot':
                resources_reshaped = padded_resources.reshape(-1, 1)
                resources_reshaped[resources_reshaped == -1] = self.num_resources
                encoded_resources = []
                for resource in resources_reshaped:
                    if resource[0] == self.num_resources:  # Padding
                        encoded_resources.append(np.zeros(self.num_resources))
                    else:
                        encoded_resources.append(
                            self.resource_encoder.transform(resource.reshape(1, -1))[0]
                        )
                encoded_resources = np.vstack(encoded_resources)
                features.extend(encoded_resources.flatten())
            else:
                normalized_resources = padded_resources.copy()
                normalized_resources[normalized_resources == -1] = 0
                normalized_resources = normalized_resources / self.num_resources
                features.extend(normalized_resources)
        
        # Add time features if used
        if self.use_time:
            if 'timestamp' in seq_df.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(seq_df['timestamp']):
                    seq_df['timestamp'] = pd.to_datetime(seq_df['timestamp'])
                
                # Calculate time differences between events
                times = seq_df['timestamp'].values
                if len(times) > 1:
                    time_diffs = np.diff(times) / np.timedelta64(1, 'h')  # Hours
                    time_diffs = np.append(time_diffs, 0)  # Add 0 for last event
                else:
                    time_diffs = np.array([0])
                
                # Pad/truncate
                if len(time_diffs) < self.max_seq_len:
                    padded_times = np.pad(
                        time_diffs, 
                        (self.max_seq_len - len(time_diffs), 0),
                        'constant', 
                        constant_values=0
                    )
                else:
                    padded_times = time_diffs[-self.max_seq_len:]
                
                # Normalize time features
                max_time = 24.0  # Cap at 24 hours for normalization
                normalized_times = np.clip(padded_times, 0, max_time) / max_time
                features.extend(normalized_times)
        
        # Add additional attributes if used
        if self.use_attributes and 'amount' in seq_df.columns:
            attributes = seq_df['amount'].values
            
            # Pad/truncate
            if len(attributes) < self.max_seq_len:
                padded_attrs = np.pad(
                    attributes, 
                    (self.max_seq_len - len(attributes), 0),
                    'constant', 
                    constant_values=0
                )
            else:
                padded_attrs = attributes[-self.max_seq_len:]
            
            # Normalize (simple min-max assuming non-negative values)
            max_attr = max(np.max(padded_attrs), 1.0)
            normalized_attrs = padded_attrs / max_attr
            features.extend(normalized_attrs)
        
        # Add case-level features
        features.append(len(seq_df) / self.max_seq_len)  # Normalized sequence length
        
        return np.array(features)


#------------------------------------------------------------------------------
# Pytorch Dataset for Sequence Models
#------------------------------------------------------------------------------

class ProcessSequenceDataset(Dataset):
    """
    Dataset for sequence models (LSTM, GRU) from process data
    """
    def __init__(self, df, max_seq_len=10, for_training=True):
        """
        Initialize dataset
        
        Args:
            df: Process data dataframe
            max_seq_len: Maximum sequence length to consider
            for_training: Whether this dataset is for training
        """
        self.df = df
        self.max_seq_len = max_seq_len
        self.for_training = for_training
        
        # Extract task and resource IDs
        self.num_tasks = df['task_id'].max() + 1
        self.num_resources = df['resource_id'].max() + 1
        
        # Process sequences
        self.sequences = []
        self._process_sequences()
    
    def _process_sequences(self):
        """Process sequences from dataframe"""
        # Group by case
        case_groups = self.df.groupby('case_id')
        
        # Process each case
        for case_id, case_data in case_groups:
            # Sort by timestamp
            case_data = case_data.sort_values('timestamp')
            
            # Skip cases with only one event (no next task)
            if len(case_data) <= 1:
                continue
            
            # Extract task sequence
            task_seq = case_data['task_id'].values
            
            # Extract other features
            resource_seq = case_data['resource_id'].values
            
            # Get timestamps if available
            if 'timestamp' in case_data.columns:
                timestamps = pd.to_datetime(case_data['timestamp']).values
                time_diffs = np.zeros_like(task_seq, dtype=np.float32)
                if len(timestamps) > 1:
                    # Calculate time differences in hours
                    time_diffs[1:] = np.diff(timestamps) / np.timedelta64(1, 'h')
            else:
                time_diffs = np.zeros_like(task_seq, dtype=np.float32)
            
            # Extract amount if available
            if 'amount' in case_data.columns:
                amounts = case_data['amount'].values
            else:
                amounts = np.zeros_like(task_seq, dtype=np.float32)
            
            # Process sub-sequences within the case
            for i in range(len(task_seq) - 1):
                # Extract prefix sequence (up to max_seq_len)
                start_idx = max(0, i + 1 - self.max_seq_len)
                prefix_tasks = task_seq[start_idx:i+1]
                prefix_resources = resource_seq[start_idx:i+1]
                prefix_times = time_diffs[start_idx:i+1]
                prefix_amounts = amounts[start_idx:i+1]
                
                # Get next task as target
                next_task = task_seq[i+1]
                
                # Store sequence
                self.sequences.append({
                    'tasks': prefix_tasks,
                    'resources': prefix_resources,
                    'times': prefix_times,
                    'amounts': prefix_amounts,
                    'next_task': next_task,
                    'length': len(prefix_tasks)
                })
    
    def __len__(self):
        """Get dataset length"""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get item by index"""
        seq = self.sequences[idx]
        
        # Create padded task sequence (right-padded with 0)
        padded_tasks = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_resources = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_times = np.zeros(self.max_seq_len, dtype=np.float32)
        padded_amounts = np.zeros(self.max_seq_len, dtype=np.float32)
        
        # Fill with sequence data (right-aligned)
        # This ensures that the most recent events are always at the end
        length = seq['length']
        offset = self.max_seq_len - length
        padded_tasks[offset:] = seq['tasks']
        padded_resources[offset:] = seq['resources']
        padded_times[offset:] = seq['times']
        padded_amounts[offset:] = seq['amounts']
        
        # Convert to tensors
        tasks_tensor = torch.tensor(padded_tasks, dtype=torch.long)
        resources_tensor = torch.tensor(padded_resources, dtype=torch.long)
        times_tensor = torch.tensor(padded_times, dtype=torch.float)
        amounts_tensor = torch.tensor(padded_amounts, dtype=torch.float)
        length_tensor = torch.tensor(length, dtype=torch.long)
        next_task_tensor = torch.tensor(seq['next_task'], dtype=torch.long)
        
        # Return dictionary of tensors
        return {
            'tasks': tasks_tensor,
            'resources': resources_tensor,
            'times': times_tensor,
            'amounts': amounts_tensor,
            'length': length_tensor,
            'next_task': next_task_tensor
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable length sequences
        
        Args:
            batch: List of samples
            
        Returns:
            Dictionary of batched tensors
        """
        tasks = torch.stack([item['tasks'] for item in batch])
        resources = torch.stack([item['resources'] for item in batch])
        times = torch.stack([item['times'] for item in batch])
        amounts = torch.stack([item['amounts'] for item in batch])
        lengths = torch.stack([item['length'] for item in batch])
        next_tasks = torch.stack([item['next_task'] for item in batch])
        
        # Create feature tensor combining all inputs
        # Shape: [batch_size, seq_len, features]
        # features = tasks + resources + times + amounts
        
        return {
            'tasks': tasks,
            'resources': resources,
            'times': times,
            'amounts': amounts,
            'lengths': lengths,
            'next_tasks': next_tasks
        }


#------------------------------------------------------------------------------
# Scikit-learn Based Models
#------------------------------------------------------------------------------

class DecisionTreeModel:
    """
    Decision Tree model for process mining
    """
    def __init__(self, max_depth=10, min_samples_split=5, criterion='gini',
                 class_weight='balanced', **kwargs):
        """
        Initialize Decision Tree model
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            criterion: Split criterion ('gini' or 'entropy')
            class_weight: Class weights ('balanced' or None)
            **kwargs: Additional parameters for DecisionTreeClassifier
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
            **kwargs
        )
        
        # Feature extractor
        self.feature_extractor = None
        self.is_fitted = False
    
    def fit(self, df, max_seq_len=10, **kwargs):
        """
        Fit model to data
        
        Args:
            df: Process data dataframe
            max_seq_len: Maximum sequence length for feature extraction
            **kwargs: Additional parameters for fit
            
        Returns:
            self
        """
        # Create feature extractor
        self.feature_extractor = ProcessFeatureExtractor(
            max_seq_len=max_seq_len,
            categorical_encoding='onehot'
        )
        
        # Fit feature extractor
        self.feature_extractor.fit(df)
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Fit model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        fit_time = time.time() - start_time
        
        logger.info(f"Fit decision tree in {fit_time:.2f} seconds")
        
        self.is_fitted = True
        return self
    
    def predict(self, df):
        """
        Make predictions
        
        Args:
            df: Process data dataframe
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X)
        pred_time = time.time() - start_time
        
        logger.info(f"Made predictions in {pred_time:.2f} seconds")
        
        return y_pred
    
    def evaluate(self, df):
        """
        Evaluate model on data
        
        Args:
            df: Process data dataframe
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before evaluation")
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        mcc = matthews_corrcoef(y, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc
        }
        
        return metrics
    
    def get_feature_importances(self):
        """
        Get feature importances
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before getting feature importances")
        
        # Get feature importances from model
        importances = self.model.feature_importances_
        
        # Create feature names (simplified for now)
        feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create dictionary
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = {k: v for k, v in sorted(
            importance_dict.items(), key=lambda item: item[1], reverse=True
        )}
        
        return importance_dict


class RandomForestModel:
    """
    Random Forest model for process mining
    """
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5,
                 criterion='gini', class_weight='balanced', n_jobs=-1, **kwargs):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            criterion: Split criterion ('gini' or 'entropy')
            class_weight: Class weights ('balanced' or None)
            n_jobs: Number of parallel jobs (-1 for all)
            **kwargs: Additional parameters for RandomForestClassifier
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
            n_jobs=n_jobs,
            **kwargs
        )
        
        # Feature extractor
        self.feature_extractor = None
        self.is_fitted = False
    
    def fit(self, df, max_seq_len=10, **kwargs):
        """
        Fit model to data
        
        Args:
            df: Process data dataframe
            max_seq_len: Maximum sequence length for feature extraction
            **kwargs: Additional parameters for fit
            
        Returns:
            self
        """
        # Create feature extractor
        self.feature_extractor = ProcessFeatureExtractor(
            max_seq_len=max_seq_len,
            categorical_encoding='onehot'
        )
        
        # Fit feature extractor
        self.feature_extractor.fit(df)
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Fit model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        fit_time = time.time() - start_time
        
        logger.info(f"Fit random forest in {fit_time:.2f} seconds")
        
        self.is_fitted = True
        return self
    
    def predict(self, df):
        """
        Make predictions
        
        Args:
            df: Process data dataframe
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X)
        pred_time = time.time() - start_time
        
        logger.info(f"Made predictions in {pred_time:.2f} seconds")
        
        return y_pred
    
    def evaluate(self, df):
        """
        Evaluate model on data
        
        Args:
            df: Process data dataframe
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before evaluation")
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        mcc = matthews_corrcoef(y, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc
        }
        
        return metrics
    
    def get_feature_importances(self):
        """
        Get feature importances
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before getting feature importances")
        
        # Get feature importances from model
        importances = self.model.feature_importances_
        
        # Create feature names (simplified for now)
        feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create dictionary
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = {k: v for k, v in sorted(
            importance_dict.items(), key=lambda item: item[1], reverse=True
        )}
        
        return importance_dict


class XGBoostModel:
    """
    XGBoost model for process mining
    """
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 objective='multi:softmax', **kwargs):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            objective: Objective function ('multi:softmax' for classification)
            **kwargs: Additional parameters for XGBoost
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': objective,
            **kwargs
        }
        
        self.model = None
        self.feature_extractor = None
        self.is_fitted = False
        self.num_classes = 0
    
    def fit(self, df, max_seq_len=10, **kwargs):
        """
        Fit model to data
        
        Args:
            df: Process data dataframe
            max_seq_len: Maximum sequence length for feature extraction
            **kwargs: Additional parameters for fit
            
        Returns:
            self
        """
        # Create feature extractor
        self.feature_extractor = ProcessFeatureExtractor(
            max_seq_len=max_seq_len,
            categorical_encoding='onehot'
        )
        
        # Fit feature extractor
        self.feature_extractor.fit(df)
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Get number of classes
        self.num_classes = len(np.unique(y))
        
        # Configure model
        params = self.params.copy()
        if 'multi' in params['objective']:
            params['num_class'] = self.num_classes
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Fit model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        fit_time = time.time() - start_time
        
        logger.info(f"Fit XGBoost in {fit_time:.2f} seconds")
        
        self.is_fitted = True
        return self
    
    def predict(self, df):
        """
        Make predictions
        
        Args:
            df: Process data dataframe
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X)
        pred_time = time.time() - start_time
        
        logger.info(f"Made predictions in {pred_time:.2f} seconds")
        
        return y_pred
    
    def evaluate(self, df):
        """
        Evaluate model on data
        
        Args:
            df: Process data dataframe
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before evaluation")
        
        # Transform data
        X, y = self.feature_extractor.transform(df)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average='macro')
        f1_weighted = f1_score(y, y_pred, average='weighted')
        mcc = matthews_corrcoef(y, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc
        }
        
        return metrics
    
    def get_feature_importances(self):
        """
        Get feature importances
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before getting feature importances")
        
        # Get feature importances from model
        importances = self.model.feature_importances_
        
        # Create feature names (simplified for now)
        feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create dictionary
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        importance_dict = {k: v for k, v in sorted(
            importance_dict.items(), key=lambda item: item[1], reverse=True
        )}
        
        return importance_dict


#------------------------------------------------------------------------------
# PyTorch Based Models
#------------------------------------------------------------------------------

class BasicLSTM(nn.Module):
    """
    Basic LSTM model for process mining
    """
    def __init__(self, num_tasks, num_resources, embedding_dim=64, hidden_dim=64,
                 num_layers=1, dropout=0.3, use_resources=True, use_time=True,
                 use_amounts=True):
        """
        Initialize LSTM model
        
        Args:
            num_tasks: Number of unique tasks
            num_resources: Number of unique resources
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_resources: Whether to use resource information
            use_time: Whether to use time features
            use_amounts: Whether to use amount features
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.num_resources = num_resources
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_resources = use_resources
        self.use_time = use_time
        self.use_amounts = use_amounts
        
        # Task embedding layer
        self.task_embedding = nn.Embedding(num_tasks + 1, embedding_dim, padding_idx=0)
        
        # Resource embedding layer
        if use_resources:
            self.resource_embedding = nn.Embedding(num_resources + 1, embedding_dim, padding_idx=0)
        
        # Calculate input dimension
        input_dim = embedding_dim
        if use_resources:
            input_dim += embedding_dim
        if use_time:
            input_dim += 1  # Time difference feature
        if use_amounts:
            input_dim += 1  # Amount feature
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_tasks)
    
    def forward(self, tasks, resources=None, times=None, amounts=None, lengths=None):
        """
        Forward pass
        
        Args:
            tasks: Task sequence tensor [batch_size, seq_len]
            resources: Resource sequence tensor [batch_size, seq_len]
            times: Time difference tensor [batch_size, seq_len]
            amounts: Amount tensor [batch_size, seq_len]
            lengths: Sequence lengths [batch_size]
            
        Returns:
            Output logits [batch_size, num_tasks]
        """
        batch_size, seq_len = tasks.size()
        
        # Embed tasks
        task_embeds = self.task_embedding(tasks)  # [batch_size, seq_len, embedding_dim]
        
        # Initialize input features
        features = [task_embeds]
        
        # Add resource embeddings if used
        if self.use_resources and resources is not None:
            resource_embeds = self.resource_embedding(resources)
            features.append(resource_embeds)
        
        # Add time features if used
        if self.use_time and times is not None:
            # Reshape to match embedding dimensions
            time_features = times.unsqueeze(2)  # [batch_size, seq_len, 1]
            features.append(time_features)
        
        # Add amount features if used
        if self.use_amounts and amounts is not None:
            # Reshape to match embedding dimensions
            amount_features = amounts.unsqueeze(2)  # [batch_size, seq_len, 1]
            features.append(amount_features)
        
        # Combine features
        x = torch.cat(features, dim=2)  # [batch_size, seq_len, input_dim]
        
        # Pack sequence for LSTM if lengths provided
        if lengths is not None:
            # Sort sequences by length for more efficient packing
            lengths_sorted, perm_idx = lengths.sort(0, descending=True)
            x_sorted = x[perm_idx]
            
            # Pack sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted, batch_first=True
            )
            
            # Apply LSTM
            output_packed, (h_n, c_n) = self.lstm(x_packed)
            
            # Unpack output
            output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
            
            # Unsort sequences
            _, unperm_idx = perm_idx.sort(0)
            output = output[unperm_idx]
            h_n = h_n[:, unperm_idx]
        else:
            # Regular LSTM
            output, (h_n, c_n) = self.lstm(x)
        
        # Get last hidden state
        last_hidden = h_n[-1]  # [batch_size, hidden_dim]
        
        # Apply dropout
        last_hidden = self.dropout_layer(last_hidden)
        
        # Apply output layer
        logits = self.fc(last_hidden)
        
        return logits
    
    def predict(self, batch):
        """
        Make prediction from batch dictionary
        
        Args:
            batch: Dictionary of input tensors
            
        Returns:
            Predictions and probabilities
        """
        # Extract inputs from batch
        tasks = batch['tasks']
        resources = batch.get('resources')
        times = batch.get('times')
        amounts = batch.get('amounts')
        lengths = batch.get('lengths')
        
        # Forward pass
        logits = self.forward(tasks, resources, times, amounts, lengths)
        
        # Get predictions
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def calculate_loss(self, batch):
        """
        Calculate loss from batch dictionary
        
        Args:
            batch: Dictionary of input tensors
            
        Returns:
            Loss tensor
        """
        # Extract inputs from batch
        tasks = batch['tasks']
        resources = batch.get('resources')
        times = batch.get('times')
        amounts = batch.get('amounts')
        lengths = batch.get('lengths')
        next_tasks = batch['next_tasks']
        
        # Forward pass
        logits = self.forward(tasks, resources, times, amounts, lengths)
        
        # Calculate loss
        loss = F.cross_entropy(logits, next_tasks)
        
        return loss


class BasicMLP(nn.Module):
    """
    Basic MLP model for process mining
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3,
                 activation=F.relu):
        """
        Initialize MLP model
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output logits [batch_size, output_dim]
        """
        # Apply hidden layers
        for i in range(0, len(self.hidden_layers), 3):
            x = self.hidden_layers[i](x)  # Linear
            x = self.hidden_layers[i+1](x)  # BatchNorm
            x = self.activation(x)  # Activation
            x = self.hidden_layers[i+2](x)  # Dropout
        
        # Apply output layer
        logits = self.output_layer(x)
        
        return logits


#------------------------------------------------------------------------------
# Model Factory
#------------------------------------------------------------------------------

def create_model(model_type, **kwargs):
    """
    Factory function to create model by type
    
    Args:
        model_type: Model type string
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    if model_type == 'decision_tree':
        return DecisionTreeModel(**kwargs)
    elif model_type == 'random_forest':
        return RandomForestModel(**kwargs)
    elif model_type == 'xgboost':
        return XGBoostModel(**kwargs)
    elif model_type == 'lstm':
        return BasicLSTM(**kwargs)
    elif model_type == 'mlp':
        return BasicMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


#------------------------------------------------------------------------------
# Example Usage
#------------------------------------------------------------------------------

def example_usage():
    """Example usage of baseline models"""
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    num_cases = 100
    num_tasks = 10
    num_resources = 5
    
    # Generate case IDs
    case_ids = np.repeat(np.arange(num_cases), np.random.randint(2, 10, size=num_cases))
    
    # Generate task IDs
    task_ids = np.random.randint(0, num_tasks, size=len(case_ids))
    
    # Generate resource IDs
    resource_ids = np.random.randint(0, num_resources, size=len(case_ids))
    
    # Generate timestamps
    start_time = pd.Timestamp('2023-01-01')
    timestamps = []
    current_time = start_time
    
    for case_id in np.unique(case_ids):
        case_mask = case_ids == case_id
        num_events = np.sum(case_mask)
        
        # Random case start time
        case_start = start_time + pd.Timedelta(days=np.random.randint(0, 30))
        
        # Generate timestamps for this case
        case_times = [case_start]
        for _ in range(1, num_events):
            # Random time increment
            increment = pd.Timedelta(hours=np.random.randint(1, 24))
            case_times.append(case_times[-1] + increment)
        
        # Assign to global list at the right positions
        timestamps.extend(case_times)
    
    # Generate amount
    amounts = np.random.uniform(10, 1000, size=len(case_ids))
    
    # Create dataframe
    df = pd.DataFrame({
        'case_id': case_ids,
        'task_id': task_ids,
        'resource_id': resource_ids,
        'timestamp': timestamps,
        'amount': amounts
    })
    
    # Create feature extractor
    feature_extractor = ProcessFeatureExtractor(max_seq_len=5)
    feature_extractor.fit(df)
    
    # Transform data for ML models
    X, y = feature_extractor.transform(df)
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Create Decision Tree model
    dt_model = DecisionTreeModel(max_depth=5)
    dt_model.fit(df)
    
    # Make predictions
    y_pred = dt_model.predict(df)
    print(f"Decision Tree predictions shape: {y_pred.shape}")
    
    # Evaluate model
    metrics = dt_model.evaluate(df)
    print(f"Decision Tree metrics: {metrics}")
    
    # Create Random Forest model
    rf_model = RandomForestModel(n_estimators=10, max_depth=5)
    rf_model.fit(df)
    
    # Evaluate Random Forest
    metrics = rf_model.evaluate(df)
    print(f"Random Forest metrics: {metrics}")
    
    # Create XGBoost model
    xgb_model = XGBoostModel(n_estimators=10, max_depth=5)
    xgb_model.fit(df)
    
    # Evaluate XGBoost
    metrics = xgb_model.evaluate(df)
    print(f"XGBoost metrics: {metrics}")
    
    # Create LSTM dataset
    lstm_dataset = ProcessSequenceDataset(df, max_seq_len=5)
    
    # Inspect dataset
    print(f"LSTM dataset size: {len(lstm_dataset)}")
    sample = lstm_dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape {v.shape}, dtype {v.dtype}")
        else:
            print(f"{k}: {v}")
    
    # Create dataloader
    batch_size = 16
    dataloader = DataLoader(
        lstm_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ProcessSequenceDataset.collate_fn
    )
    
    # Create LSTM model
    lstm_model = BasicLSTM(
        num_tasks=num_tasks,
        num_resources=num_resources,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # Mini training loop
    print("Training LSTM model for 1 epoch...")
    lstm_model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        loss = lstm_model.calculate_loss(batch)
        
        loss.backward()
        optimizer.step()
        
        print(f"Batch loss: {loss.item():.4f}")
        break  # Just one batch for example
    
    # Make predictions
    lstm_model.eval()
    with torch.no_grad():
        preds, probs = lstm_model.predict(batch)
        print(f"LSTM predictions shape: {preds.shape}")
        print(f"LSTM probabilities shape: {probs.shape}")
    
    print("Example completed successfully")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Run example
    example_usage()