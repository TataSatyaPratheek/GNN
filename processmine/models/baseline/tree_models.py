#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-based baseline models for process mining
Includes decision trees, random forests, and XGBoost models
"""

import numpy as np
import pandas as pd
import torch
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Import base model class
from processmine.models.base import BaselineModel

# Set up logging
logger = logging.getLogger(__name__)


class DecisionTreeModel(BaselineModel):
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
        super().__init__()
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
            **kwargs
        )
        self.is_fitted = False
    
    def fit(self, X, y, **kwargs):
        """
        Fit model to data
        
        Args:
            X: Feature array
            y: Target array
            **kwargs: Additional parameters for fit
            
        Returns:
            self
        """
        # Fit model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        fit_time = time.time() - start_time
        
        logger.info(f"Fit decision tree in {fit_time:.2f} seconds")
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature array
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")
        
        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X)
        pred_time = time.time() - start_time
        
        logger.debug(f"Made predictions in {pred_time:.2f} seconds")
        
        return y_pred
    
    def evaluate(self, X, y):
        """
        Evaluate model on data
        
        Args:
            X: Feature array
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
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


class RandomForestModel(BaselineModel):
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
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
            n_jobs=n_jobs,
            **kwargs
        )
        self.is_fitted = False
    
    def fit(self, X, y, **kwargs):
        """
        Fit model to data
        
        Args:
            X: Feature array 
            y: Target array
            **kwargs: Additional parameters for fit
            
        Returns:
            self
        """
        # Fit model
        start_time = time.time()
        self.model.fit(X, y, **kwargs)
        fit_time = time.time() - start_time
        
        logger.info(f"Fit random forest in {fit_time:.2f} seconds")
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature array
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")
        
        # Make predictions
        start_time = time.time()
        y_pred = self.model.predict(X)
        pred_time = time.time() - start_time
        
        logger.debug(f"Made predictions in {pred_time:.2f} seconds")
        
        return y_pred
    
    def evaluate(self, X, y):
        """
        Evaluate model on data
        
        Args:
            X: Feature array
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
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


class XGBoostModel(BaselineModel):
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
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'objective': objective,
            **kwargs
        }
        
        # Create model
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**self.params)
        except ImportError:
            logger.error("XGBoost not installed. Please install with: pip install xgboost")
            raise
        
        self.is_fitted = False
        self.num_classes = 0
        self.class_mapping = None
        self.reverse_mapping = None
    
    def fit(self, X, y, **kwargs):
        """
        Fit model to data
        
        Args:
            X: Feature array
            y: Target array
            **kwargs: Additional parameters for fit
            
        Returns:
            self
        """
        import numpy as np
        
        # Find unique classes and create a class mapping to handle gaps
        unique_classes = np.unique(y)
        self.num_classes = len(unique_classes)
        self.class_mapping = {old_cls: i for i, old_cls in enumerate(unique_classes)}
        self.reverse_mapping = {i: old_cls for old_cls, i in self.class_mapping.items()}
        
        # Map classes to contiguous integers
        y_mapped = np.array([self.class_mapping[cls_val] for cls_val in y])
        
        # Configure model
        if 'multi' in self.params['objective']:
            self.params['num_class'] = self.num_classes
            # Update model parameters
            self.model.set_params(**self.params)
        
        # Fit model with mapped classes
        start_time = time.time()
        self.model.fit(X, y_mapped, **kwargs)
        fit_time = time.time() - start_time
        
        logger.info(f"Fit XGBoost in {fit_time:.2f} seconds")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature array
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before prediction")
        
        # Make predictions
        raw_predictions = self.model.predict(X)
        
        # Map predictions back to original class labels
        if self.reverse_mapping:
            return np.array([self.reverse_mapping[p] for p in raw_predictions])
        return raw_predictions
    
    def evaluate(self, X, y):
        """
        Evaluate model on data
        
        Args:
            X: Feature array
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fit before evaluation")
        
        # Make predictions
        y_pred = self.predict(X)
        
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