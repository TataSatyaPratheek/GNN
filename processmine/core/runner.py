#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main runner for process mining analysis
"""

import os
import sys
import torch
import random
import numpy as np
import gc
import time
from colorama import init as colorama_init
from termcolor import colored
from tqdm import tqdm

# Import experiment utilities
from process_mining.core.experiment import Experiment, print_section_header

# Import memory optimization
from process_mining.utils.memory import MemoryOptimizer

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", message="invalid value encountered")

# Initialize colorama for cross-platform colored terminal output
colorama_init()

# Set random seeds for reproducibility
RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_device():
    """
    Setup computing device with fallback mechanisms
    
    Returns:
        Torch device
    """
    from process_mining.training.optimizer import setup_optimized_device
    return setup_optimized_device()

def run_analysis(data_path, **kwargs):
    """
    Run full process mining analysis
    
    Args:
        data_path: Path to dataset CSV file
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary with analysis results
    """
    # Start timing for performance tracking
    total_start_time = time.time()
    
    # Set random seeds
    set_random_seeds()
    
    # Clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create experiment
    experiment = Experiment(kwargs.get('output_dir'))
    
    # Setup device
    device = setup_device()
    
    # Track memory usage
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated")
    
    # Print configuration
    print_config(data_path, kwargs)
    
    # Load and preprocess data
    from process_mining.data.preprocessing import load_and_preprocess_data, compute_class_weights
    from process_mining.data.graph_builder import build_graph_data
    
    print_section_header("Loading and Preprocessing Data")
    df, task_encoder, resource_encoder = load_and_preprocess_data(
        data_path,
        use_adaptive_norm=kwargs.get('adaptive_norm', False),
        enhanced_features=kwargs.get('enhanced_features', False),
        enhanced_graphs=kwargs.get('enhanced_graphs', False),
        batch_size=kwargs.get('batch_size', 32)
    )
    
    # Save preprocessing info
    preproc_info = {
        "num_tasks": len(task_encoder.classes_),
        "num_resources": len(resource_encoder.classes_),
        "num_cases": df["case_id"].nunique(),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "feature_normalization": "Adaptive" if kwargs.get('adaptive_norm', False) else "Standard",
        "task_distribution": df["task_name"].value_counts().to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    experiment.save_metrics(preproc_info, "preprocessing_info.json")
    
    # Save encoded task and resource mappings
    task_mapping = {idx: name for idx, name in enumerate(task_encoder.classes_)}
    resource_mapping = {idx: name for idx, name in enumerate(resource_encoder.classes_)}
    
    mappings = {
        "task_mapping": task_mapping,
        "resource_mapping": resource_mapping
    }
    experiment.save_metrics(mappings, "feature_mappings.json")
    
    # Build graph data
    print_section_header("Building Graph Data")
    graphs = build_graph_data(df)
    
    # Setup model
    from process_mining.models.factory import setup_model
    from process_mining.training.optimizer import setup_optimizer_and_loss
    
    print_section_header(f"Setting up {kwargs.get('model_type', 'basic_gat')} Model")
    model = setup_model(
        kwargs,
        df,
        task_encoder,
        resource_encoder,
        device
    )
    
    # Compute class weights
    print(colored("📊 Computing class weights for imbalanced data...", "cyan"))
    num_classes = len(task_encoder.classes_)
    class_weights = compute_class_weights(df, num_classes)
    
    # Setup optimizer and loss
    optimizer, criterion = setup_optimizer_and_loss(model, kwargs, class_weights, device)
    
    # Run ablation study if requested
    if kwargs.get('run_ablation', False):
        from process_mining.analysis.ablation import run_ablation_study
        run_ablation_study(
            kwargs,
            df,
            graphs,
            task_encoder,
            resource_encoder,
            experiment.run_dir,
            device
        )
    
    # Train model
    from process_mining.training.trainer import train_and_evaluate_model
    
    print_section_header(f"Training {kwargs.get('model_type', 'basic_gat')} Model")
    model, metrics = train_and_evaluate_model(
        model,
        graphs,
        kwargs,
        criterion,
        optimizer,
        device,
        experiment.run_dir
    )
    
    # Run process mining analysis
    analysis_results = run_process_mining_analysis(df, task_encoder, experiment)
    
    # Run reinforcement learning if not skipped
    if not kwargs.get('skip_rl', False):
        run_reinforcement_learning(df, task_encoder, experiment)
    
    # Generate summary report
    dataset_info = {
        "filename": os.path.basename(data_path),
        "cases": df["case_id"].nunique(),
        "events": len(df),
        "activities": len(df["task_id"].unique()),
        "resources": len(df["resource_id"].unique())
    }
    
    model_results = {
        kwargs.get('model_type', 'basic_gat'): {
            "accuracy": metrics.get('accuracy', 0)
        }
    }
    
    experiment.generate_summary(dataset_info, model_results, analysis_results)
    
    # Calculate total duration
    total_duration = time.time() - total_start_time
    
    # Print final completion message
    print_section_header("Process Mining Complete")
    print(colored(f"Total Duration: {total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s", "green"))
    print(colored(f"Results saved to: {experiment.run_dir}", "green"))
    
    return {
        "run_dir": experiment.run_dir,
        "metrics": metrics,
        "analysis": analysis_results
    }

def print_config(data_path, config):
    """
    Print configuration information
    
    Args:
        data_path: Path to dataset
        config: Configuration dictionary
    """
    print("\n==== Configuration ====")
    print(f"Dataset: {data_path}")
    print(f"Model type: {config.get('model_type', 'basic_gat')}")
    print(f"Batch size: {config.get('batch_size', 32)}")
    print(f"Epochs: {config.get('epochs', 20)}")
    print(f"Feature normalization: {'Adaptive' if config.get('adaptive_norm', False) else 'Standard'}")
    print(f"Enhanced features: {config.get('enhanced_features', False)}")
    print(f"Enhanced graphs: {config.get('enhanced_graphs', False)}")
    print(f"Run ablation: {config.get('run_ablation', False)}")
    if config.get('run_ablation', False):
        print(f"Ablation type: {config.get('ablation_type', 'all')}")
    print(f"Skip RL: {config.get('skip_rl', False)}")
    print(f"Skip LSTM: {config.get('skip_lstm', False)}")

def run_process_mining_analysis(df, task_encoder, experiment):
    """
    Run process mining analysis
    
    Args:
        df: Process data dataframe
        task_encoder: Task label encoder
        experiment: Experiment instance
        
    Returns:
        Dictionary with analysis results
    """
    from process_mining.analysis.process_mining import (
        analyze_bottlenecks,
        analyze_cycle_times,
        analyze_rare_transitions,
        perform_conformance_checking,
        analyze_transition_patterns,
        spectral_cluster_graph,
        build_task_adjacency
    )
    
    print_section_header("Performing Process Mining Analysis")
    
    try:
        # Analyze bottlenecks
        print(colored("🔍 Analyzing process bottlenecks...", "cyan"))
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
        
        # Save bottleneck data
        bottleneck_stats.to_csv(os.path.join(experiment.analysis_dir, "bottleneck_stats.csv"), index=False)
        significant_bottlenecks.to_csv(os.path.join(experiment.analysis_dir, "significant_bottlenecks.csv"), index=False)
        
        # Analyze cycle times
        print(colored("🔍 Analyzing cycle times...", "cyan"))
        case_merged, long_cases, cut95 = analyze_cycle_times(df)
        
        # Save cycle time data
        case_merged.to_csv(os.path.join(experiment.analysis_dir, "case_cycle_times.csv"), index=False)
        long_cases.to_csv(os.path.join(experiment.analysis_dir, "long_running_cases.csv"), index=False)
        
        # Analyze rare transitions
        print(colored("🔍 Identifying rare transitions...", "cyan"))
        rare_trans = analyze_rare_transitions(bottleneck_stats)
        rare_trans.to_csv(os.path.join(experiment.analysis_dir, "rare_transitions.csv"), index=False)
        
        # Perform conformance checking
        print(colored("🔍 Performing conformance checking...", "cyan"))
        try:
            replayed, n_deviant = perform_conformance_checking(df)
            conformance_metrics = {
                "total_traces": len(replayed),
                "conforming_traces": len(replayed) - n_deviant,
                "deviant_traces": n_deviant,
                "conformance": float((len(replayed) - n_deviant) / len(replayed)) if replayed else 0
            }
            experiment.save_metrics(conformance_metrics, "conformance_metrics.json")
        except Exception as e:
            print(colored(f"⚠️ Conformance checking failed: {e}", "yellow"))
            print(colored("Continuing without conformance checking...", "yellow"))
            replayed, n_deviant = [], 0
        
        # Print summary
        print(colored("\n📊 Process Analysis Summary:", "cyan"))
        print(colored(f"   ✓ Found {len(significant_bottlenecks)} significant bottlenecks", "green"))
        print(colored(f"   ✓ Identified {len(long_cases)} long-running cases above 95th percentile (> {cut95:.1f}h)", "green"))
        print(colored(f"   ✓ Discovered {len(rare_trans)} rare transitions", "green"))
        print(colored(f"   ✓ Conformance Checking: {n_deviant} deviant traces out of {len(replayed) if replayed else 0}", "green"))
        
        # Create visualizations
        from process_mining.visualization.process_viz import (
            plot_cycle_time_distribution,
            plot_process_flow,
            plot_transition_heatmap,
            create_sankey_diagram
        )
        
        print_section_header("Creating Process Visualizations")
        
        # Plot cycle time distribution
        print(colored("📊 Creating cycle time distribution...", "cyan"))
        plot_cycle_time_distribution(
            case_merged["duration_h"].values,
            os.path.join(experiment.viz_dir, "cycle_time_distribution.png")
        )
        
        # Plot process flow with bottlenecks
        print(colored("📊 Creating process flow visualization...", "cyan"))
        plot_process_flow(
            bottleneck_stats, task_encoder, significant_bottlenecks.head(10),
            os.path.join(experiment.viz_dir, "process_flow_bottlenecks.png")
        )
        
        # Get transition patterns and create visualizations
        print(colored("📊 Analyzing transition patterns...", "cyan"))
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df, viz_dir=experiment.viz_dir)
        
        # Save transition data
        transitions.to_csv(os.path.join(experiment.analysis_dir, "transitions.csv"), index=False)
        
        # Plot transition heatmap
        print(colored("📊 Creating transition probability heatmap...", "cyan"))
        plot_transition_heatmap(
            transitions, task_encoder,
            os.path.join(experiment.viz_dir, "transition_probability_heatmap.png")
        )
        
        # Create Sankey diagram
        print(colored("📊 Creating process flow Sankey diagram...", "cyan"))
        create_sankey_diagram(
            transitions, task_encoder,
            os.path.join(experiment.viz_dir, "process_flow_sankey.html")
        )
        
        # Spectral clustering
        print_section_header("Performing Spectral Clustering")
        
        print(colored("🔍 Building task adjacency matrix...", "cyan"))
        num_classes = len(task_encoder.classes_)
        adj_matrix = build_task_adjacency(df, num_classes)
        
        print(colored("🔍 Performing spectral clustering...", "cyan"))
        cluster_labels = spectral_cluster_graph(adj_matrix, k=3)
        
        # Save clustering results
        clustering_results = {
            "task_clusters": {
                task_encoder.inverse_transform([t_id])[0]: int(lbl)
                for t_id, lbl in enumerate(cluster_labels)
            },
            "num_clusters": int(np.max(cluster_labels) + 1),
            "cluster_sizes": {
                f"cluster_{i}": int(np.sum(cluster_labels == i))
                for i in range(np.max(cluster_labels) + 1)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        experiment.save_metrics(clustering_results, "clustering_results.json")
        
        # Compile analysis results
        median_duration = np.median(case_merged["duration_h"].values)
        
        analysis_results = {
            "bottlenecks": len(significant_bottlenecks),
            "median_cycle_time": float(median_duration),
            "p95_cycle_time": float(cut95),
            "rare_transitions": len(rare_trans),
            "deviant_traces": n_deviant,
            "total_traces": len(replayed) if replayed else df["case_id"].nunique(),
            "num_clusters": int(np.max(cluster_labels) + 1)
        }
        
        # Save process mining analysis results
        process_analysis = {
            "num_cases": df["case_id"].nunique(),
            "num_events": len(df),
            "num_activities": len(df["task_id"].unique()),
            "num_resources": len(df["resource_id"].unique()),
            "num_long_cases": len(long_cases),
            "cycle_time_95th_percentile": float(cut95),
            "num_significant_bottlenecks": len(significant_bottlenecks),
            "num_rare_transitions": len(rare_trans),
            "num_deviant_traces": n_deviant,
            "total_traces": len(replayed) if replayed else df["case_id"].nunique(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        experiment.save_metrics(process_analysis, "process_analysis.json")
        
        return analysis_results
        
    except Exception as e:
        print(colored(f"❌ Error in process mining analysis: {e}", "red"))
        import traceback
        traceback.print_exc()
        return {}

def run_reinforcement_learning(df, task_encoder, experiment):
    """
    Run reinforcement learning for process optimization
    
    Args:
        df: Process data dataframe
        task_encoder: Task label encoder
        experiment: Experiment instance
        
    Returns:
        Dictionary with RL results
    """
    from process_mining.reinforcement.environment import ProcessEnv
    from process_mining.reinforcement.agent import run_q_learning, get_optimal_policy
    
    print_section_header("Training RL Agent")
    
    try:
        print(colored("🔄 Setting up process environment...", "cyan"))
        dummy_resources = [0, 1]  # Example with 2 resources
        env = ProcessEnv(df, task_encoder, dummy_resources)
        
        print(colored("🏋️ Training reinforcement learning agent...", "cyan"))
        rl_results = run_q_learning(
            env, 
            episodes=30,
            viz_dir=experiment.viz_dir,
            policy_dir=experiment.policy_dir
        )
        
        # Extract optimal policy
        print(colored("🔍 Extracting optimal policy...", "cyan"))
        all_actions = [(t, r) for t in env.all_tasks for r in env.resources]
        policy_results = get_optimal_policy(
            rl_results,
            all_actions,
            policy_dir=experiment.policy_dir
        )
        
        # Save policy and results
        experiment.save_metrics(policy_results, "rl_policy.json")
        
        # Generate policy summary
        total_states = len(policy_results['policy'])
        total_actions = len(all_actions)
        
        print(colored("\n📊 Reinforcement Learning Results:", "cyan"))
        print(colored(f"   ✓ Learned policy for {total_states} states", "green"))
        print(colored(f"   ✓ Action space size: {total_actions}", "green"))
        
        # Print resource distribution summary
        print(colored("\n   Resource utilization in policy:", "cyan"))
        for resource, count in sorted(policy_results['resource_distribution'].items()):
            percentage = (count / total_states) * 100
            print(colored(f"      Resource {resource}: {percentage:.1f}% ({count} states)", "green"))
        
        return policy_results
        
    except Exception as e:
        print(colored(f"❌ Error in reinforcement learning: {e}", "red"))
        import traceback
        traceback.print_exc()
        return {}

def main():
    """
    Main entry point for command-line execution
    """
    from processmine.config import parse_arguments
    
    # Parse arguments
    args = parse_arguments()
    
    # Convert arguments to dictionary
    config = vars(args)
    
    try:
        # Run analysis
        run_analysis(args.data_path, **config)
    except KeyboardInterrupt:
        print(colored("\n\n⚠️ Process mining interrupted by user", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n\n❌ Unexpected error: {e}", "red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()