"""
Factory module for creating model instances with a consistent interface.
"""

import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

def create_model(model_type: str, **kwargs) -> Any:
    """
    Factory function to create models with consistent interface
    
    Args:
        model_type: Type of model ('gnn', 'lstm', 'enhanced_gnn', 'xgboost', etc.)
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    if model_type == 'gnn':
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        return MemoryEfficientGNN(
            attention_type="basic",
            **kwargs
        )
    
    elif model_type == 'enhanced_gnn':
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        return MemoryEfficientGNN(
            attention_type="combined", 
            **kwargs
        )
    
    elif model_type == 'lstm':
        from processmine.models.sequence.lstm import NextActivityLSTM
        return NextActivityLSTM(**kwargs)
        
    elif model_type == 'enhanced_lstm':
        from processmine.models.sequence.lstm import EnhancedProcessRNN
        return EnhancedProcessRNN(**kwargs)
        
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**kwargs)
        
    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(**kwargs)
        except ImportError:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type
    
    Args:
        model_type: Model type string
        
    Returns:
        Default configuration dictionary
    """
    # Default configurations for different model types
    DEFAULT_CONFIGS = {
        'gnn': {
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'attention_type': 'basic',
            'pos_enc_dim': 16,
            'diversity_weight': 0.1,
            'pooling': 'mean',
            'predict_time': False,
            'use_batch_norm': True,
            'use_residual': True,
            'use_layer_norm': False
        },
        'enhanced_gnn': {
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'attention_type': 'combined',
            'pos_enc_dim': 16,
            'diversity_weight': 0.1,
            'pooling': 'mean',
            'predict_time': False,
            'use_batch_norm': True,
            'use_residual': True,
            'use_layer_norm': False
        },
        'lstm': {
            'hidden_dim': 64,
            'emb_dim': 64,
            'num_layers': 1,
            'dropout': 0.3,
            'bidirectional': False,
            'use_attention': True,
            'use_layer_norm': True
        },
        'enhanced_lstm': {
            'hidden_dim': 64,
            'emb_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'use_gru': False,
            'use_transformer': True,
            'num_heads': 4,
            'use_time_features': True
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
    
    return DEFAULT_CONFIGS.get(model_type, {})