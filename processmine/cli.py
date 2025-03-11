#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for ProcessMine
Main entry point for enhanced process mining with GNN, LSTM, and RL
"""

import os
import sys
import random
import numpy as np
import torch
import time
import argparse
import warnings
from datetime import datetime
from colorama import init as colorama_init
from termcolor import colored

# Initialize colorama for cross-platform colored terminal output
colorama_init()

# Import local modules
from processmine.config import add_phase1_arguments
from processmine.core.experiment import setup_results_dir, print_section_header
from processmine.core.runner import run_ablation_study, run_baseline_experiment
from processmine.utils.memory import MemoryOptimizer

# Import modularized CLI components
from processmine.cli_modules.logging_setup import setup_logging
from processmine.cli_modules.data_pipeline import run_data_pipeline
from processmine.cli_modules.model_training import run_model_training
from processmine.cli_modules.process_analysis import run_process_analysis
from processmine.cli_modules.reporting import generate_report
from processmine.cli_modules.device_setup import setup_device

def parse_arguments():
    """Parse command line arguments for ProcessMine"""
    parser = argparse.ArgumentParser(description='Enhanced Process Mining with GNN, LSTM, and RL')
    parser.add_argument('data_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for GNN training')
    parser.add_argument('--lstm-epochs', type=int, default=5, help='Number of epochs for LSTM training')
    parser.add_argument('--batch-size', type=int, default=32, help='Maximum batch size for training (will be automatically optimized)')
    parser.add_argument('--norm-features', action='store_true', help='Use L2 normalization for features')
    parser.add_argument('--skip-rl', action='store_true', help='Skip reinforcement learning step')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM modeling step')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
    
    # Add Phase 1 specific arguments
    add_phase1_arguments(parser)
    
    return parser.parse_args()

def main():
    """Main entry point for ProcessMine"""
    # Start timing for performance tracking
    total_start_time = time.time()
    
    # Set random seeds for reproducibility 
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging and suppress warnings
    logger = setup_logging(args)
    
    # Setup computing device
    device = setup_device()
    
    # Create results directory
    run_dir = setup_results_dir(args.output_dir)
    print(colored(f"\n📊 Results will be saved in: {run_dir}", "magenta"))
    
    # Define directory paths for different output types
    viz_dir = os.path.join(run_dir, "visualizations")
    models_dir = os.path.join(run_dir, "models")
    analysis_dir = os.path.join(run_dir, "analysis")
    metrics_dir = os.path.join(run_dir, "metrics")
    policy_dir = os.path.join(run_dir, "policies")
    ablation_dir = os.path.join(run_dir, "ablation")
    
    # Process data and prepare for modeling
    print_section_header("Loading and Preprocessing Data")
    data = run_data_pipeline(args, run_dir)
    df = data['df']
    graphs = data['graphs']
    task_encoder = data['task_encoder']
    resource_encoder = data['resource_encoder']
    
    # Check if we're using Phase 1 models and features
    using_phase1 = args.model_type in ['decision_tree', 'random_forest', 'xgboost', 'mlp', 
                                      'positional_gat', 'diverse_gat', 'enhanced_gnn'] or \
                  args.adaptive_norm or args.enhanced_features or args.enhanced_graphs
    
    # Run ablation study if requested
    if args.run_ablation:
        run_ablation_study(args, df, graphs, task_encoder, resource_encoder, run_dir, device)
    else:
        # Train and evaluate model
        print_section_header(f"Training and Evaluating {args.model_type} Model")
        model, metrics = run_model_training(
            args, data, device, run_dir,
            models_dir=models_dir, 
            viz_dir=viz_dir
        )
    
    # Process Mining Analysis
    print_section_header("Performing Process Mining Analysis")
    analysis_results = run_process_analysis(
        args, df, task_encoder, run_dir, 
        analysis_dir=analysis_dir, 
        viz_dir=viz_dir
    )
    
    # Reinforcement Learning (unless skipped)
    if not args.skip_rl:
        print_section_header("Training RL Agent")
        from processmine.process_mining.optimization import ProcessEnv, run_q_learning, get_optimal_policy
        
        rl_results = run_q_learning(
            ProcessEnv(df, task_encoder, [0, 1]), 
            episodes=30, 
            viz_dir=viz_dir, 
            policy_dir=policy_dir
        )
    
    # Generate final summary report
    generate_report(
        args, data, model, metrics if 'metrics' in locals() else None, 
        analysis_results, run_dir, total_start_time
    )
    
    # Print final completion message
    total_duration = time.time() - total_start_time
    print_section_header("Process Mining Complete")
    print(colored(f"Total Duration: {total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s", "green"))
    print(colored(f"Results saved to: {run_dir}", "green"))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored("\n\nProcess mining interrupted by user", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n\nUnexpected error: {e}", "red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)