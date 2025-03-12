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

# Import model factory
from processmine.models.factory import create_model, get_model_config

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