#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration module for ProcessMine
Handles command-line arguments and configuration settings
"""

import os
import argparse
import yaml
import logging
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

def add_phase1_arguments(parser):
    """
    Add Phase 1 specific command-line arguments
    
    Args:
        parser: ArgumentParser to modify
    """
    # Model selection
    model_group = parser.add_argument_group('Model Selection')
    model_group.add_argument(
        '--model-type', type=str, default='basic_gat',
        choices=[
            # Baseline models
            'decision_tree', 'random_forest', 'xgboost', 'mlp', 'lstm',
            # GNN models
            'basic_gat', 'positional_gat', 'diverse_gat', 'enhanced_gnn'
        ],
        help='Type of model to use'
    )
    
    # Feature engineering
    feat_group = parser.add_argument_group('Feature Engineering')
    feat_group.add_argument(
        '--adaptive-norm', action='store_true',
        help='Use adaptive normalization based on data characteristics'
    )
    feat_group.add_argument(
        '--enhanced-features', action='store_true',
        help='Use enhanced feature engineering'
    )
    feat_group.add_argument(
        '--enhanced-graphs', action='store_true',
        help='Use enhanced graph building with advanced edge features'
    )
    
    # Architecture settings
    arch_group = parser.add_argument_group('Model Architecture')
    arch_group.add_argument(
        '--hidden-dim', type=int, default=64,
        help='Hidden dimension for neural models'
    )
    arch_group.add_argument(
        '--num-layers', type=int, default=2,
        help='Number of layers in neural models'
    )
    arch_group.add_argument(
        '--num-heads', type=int, default=4,
        help='Number of attention heads for GAT models'
    )
    arch_group.add_argument(
        '--dropout', type=float, default=0.5,
        help='Dropout probability'
    )
    arch_group.add_argument(
        '--pos-dim', type=int, default=16,
        help='Positional encoding dimension for position-enhanced models'
    )
    arch_group.add_argument(
        '--diversity-weight', type=float, default=0.1,
        help='Weight for attention diversity loss'
    )
    arch_group.add_argument(
        '--predict-time', action='store_true',
        help='Whether to also predict cycle time (dual task)'
    )
    
    # Ablation study
    ablation_group = parser.add_argument_group('Ablation Study')
    ablation_group.add_argument(
        '--run-ablation', action='store_true',
        help='Run ablation study on model components'
    )
    ablation_group.add_argument(
        '--ablation-type', type=str, default='all',
        choices=['all', 'attention', 'position', 'diversity', 'architecture'],
        help='Type of ablation study to run'
    )
    
    # Training parameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate'
    )
    train_group.add_argument(
        '--weight-decay', type=float, default=5e-4,
        help='Weight decay for regularization'
    )

class Config:
    """Configuration class for ProcessMine"""
    
    def __init__(self, config_file: Optional[str] = None, args: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to YAML configuration file
            args: Optional dictionary of arguments
        """
        self.config = {}
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Override with args if provided
        if args:
            self._update_from_args(args)
    
    def _load_from_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file
        
        Args:
            config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            self.config = {}
    
    def _update_from_args(self, args: Dict[str, Any]) -> None:
        """
        Update configuration from arguments dictionary
        
        Args:
            args: Dictionary of arguments
        """
        # Convert argparse Namespace to dict if needed
        if hasattr(args, '__dict__'):
            args = vars(args)
        
        # Update config with arguments
        self.config.update(args)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to file
        
        Args:
            filepath: Path to save configuration
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {filepath}")
        except Exception as e:
            logger.error(f"Error saving configuration to {filepath}: {e}")

def load_config_from_args(args):
    """
    Create a Config object from command-line arguments
    
    Args:
        args: ArgumentParser parsed arguments
        
    Returns:
        Config object
    """
    # Convert args to dict
    args_dict = vars(args)
    
    # Check if a config file was specified
    config_file = args.config_file if hasattr(args, 'config_file') else None
    
    # Create and return Config object
    return Config(config_file=config_file, args=args_dict)

# Default configurations for different model types
DEFAULT_CONFIGS = {
    'basic_gat': {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.5,
        'lr': 0.001,
        'weight_decay': 5e-4
    },
    'positional_gat': {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.5,
        'pos_dim': 16,
        'lr': 0.001,
        'weight_decay': 5e-4
    },
    'diverse_gat': {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.5,
        'diversity_weight': 0.1,
        'lr': 0.001,
        'weight_decay': 5e-4
    },
    'enhanced_gnn': {
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.5,
        'pos_dim': 16,
        'diversity_weight': 0.1,
        'predict_time': False,
        'lr': 0.001,
        'weight_decay': 5e-4
    },
    'lstm': {
        'hidden_dim': 64,
        'embedding_dim': 64,
        'num_layers': 1,
        'dropout': 0.3,
        'lr': 0.001,
        'weight_decay': 1e-5
    },
    'mlp': {
        'hidden_dims': [64, 32],
        'dropout': 0.3,
        'lr': 0.001,
        'weight_decay': 1e-4
    },
    'decision_tree': {
        'max_depth': 10,
        'min_samples_split': 5,
        'criterion': 'gini',
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'criterion': 'gini',
        'class_weight': 'balanced',
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'multi:softmax'
    }
}

def get_model_config(model_type):
    """
    Get default configuration for a model type
    
    Args:
        model_type: Model type string
        
    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIGS.get(model_type, {})