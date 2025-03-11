#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core runner module for process mining
Contains high-level execution logic and workflow orchestration
"""

import os
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from processmine.core.experiment import print_section_header
from processmine.utils.ablation import AblationManager

def run_ablation_study(args, df, graphs, task_encoder, resource_encoder, run_dir, device):
    """
    Run ablation study on model components
    
    Args:
        args: Command-line arguments
        df: Process data dataframe
        graphs: List of graph data objects
        task_encoder: Task label encoder
        resource_encoder: Resource label encoder
        run_dir: Run directory path
        device: Torch device
    """
    print_section_header("Running Ablation Study")
    
    # Create output directory
    ablation_dir = os.path.join(run_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Calculate input dimension based on enhanced features
    num_features = len([col for col in df.columns if col.startswith('feat_')])
    
    # Base configuration for the model
    base_config = {
        "model_type": "enhanced_gnn",
        "input_dim": num_features,
        "num_classes": len(task_encoder.classes_),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.num_heads,
        "dropout": args.dropout,
        "pos_dim": args.pos_dim,
        "diversity_weight": args.diversity_weight,
        "predict_time": args.predict_time,
        "use_batch_norm": True,
        "lr": 0.001,
        "weight_decay": 5e-4,
        "batch_size": args.batch_size,
        "epochs": 5  # Use fewer epochs for ablation
    }
    
    # Create ablation manager
    ablation = AblationManager(base_config, ablation_dir, f"ablation_{args.ablation_type}")
    
    # Define variants based on ablation type
    if args.ablation_type == 'all' or args.ablation_type == 'attention':
        # Attention mechanism ablation
        ablation.add_variant(
            "single_head",
            {"heads": 1},
            "Single attention head (no multi-head attention)"
        )
        
        ablation.add_variant(
            "more_heads",
            {"heads": 8},
            "8 attention heads (increased expressivity)"
        )
    
    if args.ablation_type == 'all' or args.ablation_type == 'position':
        # Positional encoding ablation
        ablation.add_variant(
            "no_position",
            {"pos_dim": 0},
            "No positional encoding"
        )
        
        ablation.add_variant(
            "large_position",
            {"pos_dim": 32},
            "Larger positional encoding dimension (32)"
        )
    
    if args.ablation_type == 'all' or args.ablation_type == 'diversity':
        # Diversity mechanism ablation
        ablation.add_variant(
            "no_diversity",
            {"diversity_weight": 0.0},
            "No attention diversity regularization"
        )
        
        ablation.add_variant(
            "high_diversity",
            {"diversity_weight": 0.5},
            "Higher attention diversity weight (0.5)"
        )
    
    if args.ablation_type == 'all' or args.ablation_type == 'architecture':
        # Architecture ablation
        ablation.add_variant(
            "shallow",
            {"num_layers": 1},
            "Shallow model (1 layer)"
        )
        
        ablation.add_variant(
            "deep",
            {"num_layers": 4},
            "Deep model (4 layers)"
        )
        
        ablation.add_variant(
            "narrow",
            {"hidden_dim": 32},
            "Narrow hidden dimension (32)"
        )
        
        ablation.add_variant(
            "wide",
            {"hidden_dim": 128},
            "Wide hidden dimension (128)"
        )
    
    # Define evaluation function
    def evaluate_model_config(config):
        """
        Evaluate a model configuration during ablation study
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Metrics dictionary or error information
        """
        try:
            # Import based on model type
            from processmine.models.gnn.enhanced import create_enhanced_gnn
            
            # Create model with this config
            model = create_enhanced_gnn(
                input_dim=config["input_dim"],
                num_classes=config["num_classes"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                heads=config["heads"],
                dropout=config["dropout"],
                pos_dim=config.get("pos_dim", 16),
                diversity_weight=config.get("diversity_weight", 0.1),
                predict_time=config.get("predict_time", False),
                use_batch_norm=config.get("use_batch_norm", True)
            ).to(device)
            
            # Set up optimizer and loss
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.get("lr", 0.001),
                weight_decay=config.get("weight_decay", 5e-4)
            )
            
            if config.get("predict_time", False):
                from processmine.utils.losses import ProcessLoss
                criterion = ProcessLoss(
                    task_weight=0.7,
                    time_weight=0.3
                ).to(device)
            else:
                criterion = torch.nn.CrossEntropyLoss()
            
            # Train for a few epochs
            epochs = config.get("epochs", 5)
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    
                    outputs = model(data)
                    
                    if isinstance(outputs, dict):
                        # Handle enhanced model outputs
                        if config.get("predict_time", False):
                            # Get graph-level predictions and targets
                            task_pred = outputs["task_pred"]
                            # For PyG batches, we need to use batch to extract graph-level targets
                            graph_targets = get_graph_targets(data.y, data.batch)
                            
                            loss, _ = criterion(
                                task_pred, 
                                graph_targets,
                                time_pred=outputs.get("time_pred"), 
                                time_target=None  # No time target in synthetic data
                            )
                        else:
                            # Get graph-level predictions and targets
                            task_pred = outputs["task_pred"]
                            # For PyG batches, we need to use batch to extract graph-level targets
                            graph_targets = get_graph_targets(data.y, data.batch)
                            
                            loss = criterion(task_pred, graph_targets)
                            
                        # Add diversity loss if available
                        if "diversity_loss" in outputs and config.get("diversity_weight", 0.0) > 0:
                            loss = loss + outputs["diversity_loss"]
                    else:
                        # Simple output (just logits)
                        # Get graph-level targets
                        graph_targets = get_graph_targets(data.y, data.batch)
                        loss = criterion(outputs, graph_targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
            
            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    
                    outputs = model(data)
                    
                    if isinstance(outputs, dict):
                        # Handle enhanced model outputs
                        if config.get("predict_time", False):
                            # Get graph-level predictions and targets
                            task_pred = outputs["task_pred"]
                            graph_targets = get_graph_targets(data.y, data.batch)
                            
                            loss, _ = criterion(
                                task_pred, 
                                graph_targets,
                                time_pred=outputs.get("time_pred"), 
                                time_target=None
                            )
                            logits = task_pred
                        else:
                            # Get graph-level predictions and targets
                            task_pred = outputs["task_pred"]
                            graph_targets = get_graph_targets(data.y, data.batch)
                            
                            loss = criterion(task_pred, graph_targets)
                            logits = task_pred
                    else:
                        # Simple output (just logits)
                        logits = outputs
                        graph_targets = get_graph_targets(data.y, data.batch)
                        loss = criterion(logits, graph_targets)
                    
                    test_loss += loss.item()
                    
                    # Get predictions
                    _, predicted = torch.max(logits, 1)
                    
                    # Track true and predicted labels
                    y_true.extend(graph_targets.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    
                    # Track accuracy
                    total += graph_targets.size(0)
                    correct += (predicted == graph_targets).sum().item()
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate F1 score
            from sklearn.metrics import f1_score, precision_score, recall_score
            if len(np.unique(y_true)) > 1:  # Make sure there are at least 2 classes
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            else:
                f1_macro = f1_weighted = precision = recall = 0.0
            
            # Return metrics
            return {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision": precision,
                "recall": recall,
                "test_loss": test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
            }
            
        except Exception as e:
            print(colored(f"Error evaluating config: {e}", "red"))
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    # Helper function to get graph-level targets
    def get_graph_targets(node_targets, batch):
        """
        Convert node-level targets to graph-level targets
        
        Args:
            node_targets: Node-level target tensor
            batch: Batch assignment tensor
            
        Returns:
            Graph-level target tensor
        """
        # Use the most common target for each graph
        unique_graphs = torch.unique(batch)
        graph_targets = []
        
        for g in unique_graphs:
            # Get targets for this graph
            graph_mask = (batch == g)
            graph_node_targets = node_targets[graph_mask]
            
            # Find most common target (mode)
            if len(graph_node_targets) > 0:
                values, counts = torch.unique(graph_node_targets, return_counts=True)
                mode_idx = torch.argmax(counts)
                graph_targets.append(values[mode_idx])
            else:
                # Fallback if no targets (should not happen)
                graph_targets.append(0)
        
        return torch.tensor(graph_targets, dtype=torch.long, device=node_targets.device)
    
    # Set evaluation function
    ablation.set_evaluation_function(evaluate_model_config)
    
    # Run ablation study
    results = ablation.run()
    
    # Plot ablation results
    ablation.plot_ablation_results()
    
    # Generate report
    report_path = ablation.generate_report()
    
    print(colored(f"Ablation study completed. Report saved to: {report_path}", "green"))
    
    return results

def run_baseline_experiment(args, df, graphs, task_encoder, resource_encoder, run_dir, device):
    """
    Run standard experiment with baseline models
    
    Args:
        args: Command-line arguments
        df: Process data dataframe
        graphs: List of graph data objects
        task_encoder: Task label encoder
        resource_encoder: Resource label encoder
        run_dir: Run directory path
        device: Torch device
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    print_section_header(f"Running Experiment with {args.model_type} Model")
    
    from processmine.models.base import setup_phase1_model, setup_optimizer_and_loss
    from processmine.data.preprocessing import compute_class_weights
    from processmine.core.training import train_and_evaluate_model_phase1
    
    # Compute class weights for imbalanced data
    print(colored("📊 Computing class weights for imbalanced data...", "cyan"))
    num_classes = len(task_encoder.classes_)
    class_weights = compute_class_weights(df, num_classes)
    
    # Setup model
    model = setup_phase1_model(args, df, task_encoder, resource_encoder, device)
    
    # Setup optimizer and loss function
    optimizer, criterion = setup_optimizer_and_loss(model, args, class_weights, device)
    
    # Train and evaluate model
    model, metrics = train_and_evaluate_model_phase1(model, graphs, args, criterion, optimizer, device, run_dir)
    
    return model, metrics

def run_reinforcement_learning(df, task_encoder, run_dir, device, episodes=30):
    """
    Run reinforcement learning optimization
    
    Args:
        df: Process data dataframe
        task_encoder: Task label encoder
        run_dir: Run directory path
        device: Torch device
        episodes: Number of RL episodes
    
    Returns:
        Dictionary with RL results
    """
    print_section_header("Training RL Agent for Process Optimization")
    
    from processmine.process_mining.optimization import ProcessEnv, run_q_learning, get_optimal_policy
    
    # Setup paths
    viz_dir = os.path.join(run_dir, "visualizations")
    policy_dir = os.path.join(run_dir, "policies")
    
    # Ensure directories exist
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)
    
    # Setup environment
    print(colored("🔄 Setting up process environment...", "cyan"))
    dummy_resources = [0, 1]  # Example with 2 resources
    env = ProcessEnv(df, task_encoder, dummy_resources)
    
    # Train RL agent
    print(colored("🏋️ Training reinforcement learning agent...", "cyan"))
    rl_results = run_q_learning(env, episodes=episodes, viz_dir=viz_dir, policy_dir=policy_dir)
    
    # Extract optimal policy
    print(colored("🔍 Extracting optimal policy...", "cyan"))
    all_actions = [(t, r) for t in env.all_tasks for r in env.resources]
    policy_results = get_optimal_policy(rl_results, all_actions, policy_dir=policy_dir)
    
    # Print summary
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

def run_process_analysis(df, task_encoder, run_dir):
    """
    Run comprehensive process mining analysis
    
    Args:
        df: Process data dataframe
        task_encoder: Task label encoder
        run_dir: Run directory path
    
    Returns:
        Dictionary with analysis results
    """
    print_section_header("Performing Process Mining Analysis")
    
    from processmine.process_mining.analysis import (
        analyze_bottlenecks,
        analyze_cycle_times,
        analyze_rare_transitions,
        perform_conformance_checking,
        analyze_transition_patterns,
        spectral_cluster_graph,
        build_task_adjacency
    )
    from processmine.visualization.process_viz import (
        plot_cycle_time_distribution,
        plot_process_flow,
        plot_transition_heatmap,
        create_sankey_diagram
    )
    from processmine.core.experiment import save_metrics
    
    # Setup paths
    analysis_dir = os.path.join(run_dir, "analysis")
    viz_dir = os.path.join(run_dir, "visualizations")
    
    # Ensure directories exist
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    results = {}
    
    try:
        # Analyze bottlenecks
        print(colored("🔍 Analyzing process bottlenecks...", "cyan"))
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
        
        # Save bottleneck data
        bottleneck_stats.to_csv(os.path.join(analysis_dir, "bottleneck_stats.csv"), index=False)
        significant_bottlenecks.to_csv(os.path.join(analysis_dir, "significant_bottlenecks.csv"), index=False)
        
        # Store in results
        results['bottlenecks'] = {
            'total': len(bottleneck_stats),
            'significant': len(significant_bottlenecks)
        }
        
        # Analyze cycle times
        print(colored("🔍 Analyzing cycle times...", "cyan"))
        case_merged, long_cases, cut95 = analyze_cycle_times(df)
        
        # Save cycle time data
        case_merged.to_csv(os.path.join(analysis_dir, "case_cycle_times.csv"), index=False)
        long_cases.to_csv(os.path.join(analysis_dir, "long_running_cases.csv"), index=False)
        
        # Store in results
        results['cycle_times'] = {
            'median': float(np.median(case_merged["duration_h"])),
            'mean': float(np.mean(case_merged["duration_h"])),
            'p95': float(cut95),
            'long_cases': len(long_cases)
        }
        
        # Plot cycle time distribution
        plot_cycle_time_distribution(
            case_merged["duration_h"].values,
            os.path.join(viz_dir, "cycle_time_distribution.png")
        )
        
        # Analyze rare transitions
        print(colored("🔍 Identifying rare transitions...", "cyan"))
        rare_trans = analyze_rare_transitions(bottleneck_stats)
        rare_trans.to_csv(os.path.join(analysis_dir, "rare_transitions.csv"), index=False)
        
        # Store in results
        results['rare_transitions'] = {
            'count': len(rare_trans)
        }
        
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
            save_metrics(conformance_metrics, run_dir, "conformance_metrics.json")
            
            # Store in results
            results['conformance'] = conformance_metrics
            
        except Exception as e:
            print(colored(f"⚠️ Conformance checking failed: {e}", "yellow"))
            results['conformance'] = {'error': str(e)}
        
        # Plot process flow with bottlenecks
        print(colored("📊 Creating process flow visualization...", "cyan"))
        plot_process_flow(
            bottleneck_stats, task_encoder, significant_bottlenecks.head(10),
            os.path.join(viz_dir, "process_flow_bottlenecks.png")
        )
        
        # Get transition patterns and create visualizations
        print(colored("📊 Analyzing transition patterns...", "cyan"))
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df, viz_dir=viz_dir)
        
        # Save transition data
        transitions.to_csv(os.path.join(analysis_dir, "transitions.csv"), index=False)
        
        # Plot transition heatmap
        print(colored("📊 Creating transition probability heatmap...", "cyan"))
        plot_transition_heatmap(
            transitions, task_encoder,
            os.path.join(viz_dir, "transition_probability_heatmap.png")
        )
        
        # Create Sankey diagram
        print(colored("📊 Creating process flow Sankey diagram...", "cyan"))
        create_sankey_diagram(
            transitions, task_encoder,
            os.path.join(viz_dir, "process_flow_sankey.html")
        )
        
        # Store in results
        results['transitions'] = {
            'count': len(transitions),
            'unique': len(trans_count)
        }
        
        # Spectral clustering
        print(colored("🔍 Performing spectral clustering...", "cyan"))
        num_classes = len(task_encoder.classes_)
        adj_matrix = build_task_adjacency(df, num_classes)
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_metrics(clustering_results, run_dir, "clustering_results.json")
        
        # Store in results
        results['clustering'] = {
            'num_clusters': int(np.max(cluster_labels) + 1),
            'cluster_sizes': clustering_results['cluster_sizes']
        }
        
        # Overall process summary
        process_summary = {
            "num_cases": df["case_id"].nunique(),
            "num_events": len(df),
            "num_activities": len(df["task_id"].unique()),
            "num_resources": len(df["resource_id"].unique()),
            "num_long_cases": len(long_cases),
            "cycle_time_95th_percentile": float(cut95),
            "num_significant_bottlenecks": len(significant_bottlenecks),
            "num_rare_transitions": len(rare_trans),
            "num_deviant_traces": n_deviant if 'n_deviant' in locals() else 0,
            "total_traces": len(replayed) if 'replayed' in locals() and replayed else df["case_id"].nunique()
        }
        save_metrics(process_summary, run_dir, "process_analysis.json")
        
        # Store in results
        results['summary'] = process_summary
        
    except Exception as e:
        print(colored(f"❌ Error in process mining analysis: {e}", "red"))
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    return results