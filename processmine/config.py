#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration utilities for process mining
"""

import argparse

def create_argument_parser():
    """
    Create the argument parser for the process mining CLI
    
    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description='Enhanced Process Mining with GNN, LSTM, and RL')
    parser.add_argument('data_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for GNN training')
    parser.add_argument('--lstm-epochs', type=int, default=5, help='Number of epochs for LSTM training')
    parser.add_argument('--batch-size', type=int, default=32, help='Maximum batch size for training (will be automatically optimized)')
    parser.add_argument('--norm-features', action='store_true', help='Use L2 normalization for features')
    parser.add_argument('--skip-rl', action='store_true', help='Skip reinforcement learning step')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM modeling step')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
    
    # Add model selection arguments
    add_model_arguments(parser)
    
    return parser

def add_model_arguments(parser):
    """
    Add model-specific command-line arguments
    
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

def parse_arguments(args=None):
    """
    Parse command-line arguments
    
    Args:
        args: Optional list of arguments to parse (if None, uses sys.argv)
        
    Returns:
        Parsed arguments namespace
    """
    parser = create_argument_parser()
    return parser.parse_args(args)