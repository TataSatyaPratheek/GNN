# Main processmine/__init__.py
"""
ProcessMine: Memory-efficient process mining with Graph Neural Networks and LSTM models

A comprehensive toolkit for process mining using advanced machine learning approaches
including Graph Neural Networks, LSTMs, and Reinforcement Learning.
All components are highly optimized for memory efficiency and performance.
"""

__version__ = '0.1.0'

# Import main components for easier access
from processmine.core.runner import run_analysis, run_training, run_optimization
from processmine.data.loader import load_and_preprocess_data
from processmine.utils.memory import log_memory_usage, clear_memory, get_memory_stats

# Import for model creation
def create_model(model_type, **kwargs):
    """
    Factory function to create models of different types with consistent interface
    
    Args:
        model_type: Type of model to create ('gnn', 'lstm', 'enhanced_gnn', etc.)
        **kwargs: Model-specific parameters
        
    Returns:
        Instantiated model
    """
    if model_type == 'gnn':
        from processmine.models.gnn.architectures import OptimizedGNN
        return OptimizedGNN(
            attention_type="basic",
            **kwargs
        )
    
    elif model_type == 'enhanced_gnn':
        from processmine.models.gnn.architectures import OptimizedGNN
        return OptimizedGNN(
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

# Process Mining Core
from processmine.process_mining.analysis import (
    analyze_bottlenecks,
    analyze_cycle_times,
    analyze_transition_patterns,
    identify_process_variants,
    analyze_resource_workload
)

# Visualization
from processmine.visualization.viz import ProcessVisualizer