#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for enhanced process mining with GNN, LSTM, and RL
"""

import os
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from datetime import datetime
import json
import sys
import shutil

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# Import local modules
from modules.data_preprocessing import (
    load_and_preprocess_data,
    create_feature_representation,
    build_graph_data,
    compute_class_weights
)
from models.gat_model import (
    NextTaskGAT,
    train_gat_model,
    evaluate_gat_model
)
from models.lstm_model import (
    NextActivityLSTM,
    prepare_sequence_data,
    make_padded_dataset,
    train_lstm_model,
    evaluate_lstm_model
)
from modules.process_mining import (
    analyze_bottlenecks,
    analyze_cycle_times,
    analyze_rare_transitions,
    perform_conformance_checking,
    analyze_transition_patterns,
    spectral_cluster_graph,
    build_task_adjacency
)
from modules.rl_optimization import (
    ProcessEnv,
    run_q_learning,
    get_optimal_policy
)
from visualization.process_viz import (
    plot_confusion_matrix,
    plot_embeddings,
    plot_cycle_time_distribution,
    plot_process_flow,
    plot_transition_heatmap,
    create_sankey_diagram
)

def setup_results_dir():
    """Create timestamped results directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "results")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create subdirectories
    subdirs = [
        "models",          # For saved model weights
        "visualizations",  # For all plots and diagrams
        "metrics",        # For performance metrics
        "analysis",       # For process mining analysis results
        "policies"        # For RL policies
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir

def save_metrics(metrics_dict, run_dir, filename):
    """Save metrics to JSON file"""
    filepath = os.path.join(run_dir, "metrics", filename)
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def main():
    # Create results directory
    run_dir = setup_results_dir()
    print(f"Results will be saved in: {run_dir}")
    
    # 1. Load and preprocess data
    if len(sys.argv) < 2:
        raise ValueError("Error: Missing dataset path. Please provide the path to the dataset as a command line argument.")
    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at {data_path}")
        return
    
    print("\n1. Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    df, le_task, le_resource = create_feature_representation(df, use_norm_features=True)
    
    # Save preprocessing info
    preproc_info = {
        "num_tasks": len(le_task.classes_),
        "num_resources": len(le_resource.classes_),
        "num_cases": df["case_id"].nunique(),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())]
    }
    save_metrics(preproc_info, run_dir, "preprocessing_info.json")
    
    # 2. Build graph data
    print("\n2. Building graph data...")
    graphs = build_graph_data(df)
    train_size = int(len(graphs)*0.8)
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    
    # 3. Train GAT model
    print("\n3. Training GAT model...")
    num_classes = len(le_task.classes_)
    class_weights = compute_class_weights(df, num_classes).to(device)
    
    gat_model = NextTaskGAT(5, 64, num_classes, num_layers=2, heads=4, dropout=0.5).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(gat_model.parameters(), lr=0.0005, weight_decay=5e-4)
    
    gat_model_path = os.path.join(run_dir, "models", "best_gnn_model.pth")
    gat_model = train_gat_model(
        gat_model, train_loader, val_loader,
        criterion, optimizer, device,
        num_epochs=20, model_path=gat_model_path
    )
    
    # 4. Evaluate GAT model
    print("\n4. Evaluating GAT model...")
    y_true, y_pred, y_prob = evaluate_gat_model(gat_model, val_loader, device)
    plot_confusion_matrix(
        y_true, y_pred, le_task.classes_,
        os.path.join(run_dir, "visualizations", "gat_confusion_matrix.png")
    )
    
    # Save GAT metrics
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    gat_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
    }
    save_metrics(gat_metrics, run_dir, "gat_metrics.json")
    
    # 5. Train LSTM model
    print("\n5. Training LSTM model...")
    train_seq, test_seq = prepare_sequence_data(df)
    X_train_pad, X_train_len, y_train_lstm, _ = make_padded_dataset(train_seq, num_classes)
    X_test_pad, X_test_len, y_test_lstm, _ = make_padded_dataset(test_seq, num_classes)
    
    lstm_model = NextActivityLSTM(num_classes, emb_dim=64, hidden_dim=64, num_layers=1).to(device)
    lstm_model_path = os.path.join(run_dir, "models", "lstm_next_activity.pth")
    lstm_model = train_lstm_model(
        lstm_model, X_train_pad, X_train_len, y_train_lstm,
        device, batch_size=64, epochs=5, model_path=lstm_model_path
    )
    
    # 6. Process Mining Analysis
    print("\n6. Performing process mining analysis...")
    bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
    case_merged, long_cases, cut95 = analyze_cycle_times(df)
    rare_trans = analyze_rare_transitions(bottleneck_stats)
    replayed, n_deviant = perform_conformance_checking(df)
    
    # Save process mining analysis results
    process_analysis = {
        "num_long_cases": len(long_cases),
        "cycle_time_95th_percentile": float(cut95),
        "num_rare_transitions": len(rare_trans),
        "num_deviant_traces": n_deviant,
        "total_traces": len(replayed)
    }
    save_metrics(process_analysis, run_dir, "process_analysis.json")
    
    print(f"Found {len(long_cases)} long-running cases above 95th percentile (> {cut95:.1f}h)")
    print(f"Found {len(rare_trans)} rare transitions")
    print(f"Conformance Checking: {n_deviant} deviant traces out of {len(replayed)}")
    
    # 7. Visualizations
    print("\n7. Creating visualizations...")
    viz_dir = os.path.join(run_dir, "visualizations")
    plot_cycle_time_distribution(
        case_merged["duration_h"].values,
        os.path.join(viz_dir, "cycle_time_distribution.png")
    )
    plot_process_flow(
        bottleneck_stats, le_task, significant_bottlenecks.head(),
        os.path.join(viz_dir, "process_flow_bottlenecks.png")
    )
    
    # Get transition patterns first
    transitions, trans_count, prob_matrix = analyze_transition_patterns(df)
    plot_transition_heatmap(
        transitions, le_task,
        os.path.join(viz_dir, "transition_probability_heatmap.png")
    )
    create_sankey_diagram(
        transitions, le_task,
        os.path.join(viz_dir, "process_flow_sankey.html")
    )
    
    # 8. Spectral Clustering
    print("\n8. Performing spectral clustering...")
    adj_matrix = build_task_adjacency(df, num_classes)
    cluster_labels = spectral_cluster_graph(adj_matrix, k=3)
    
    # Save clustering results
    clustering_results = {
        "task_clusters": {
            le_task.inverse_transform([t_id])[0]: int(lbl)
            for t_id, lbl in enumerate(cluster_labels)
        }
    }
    save_metrics(clustering_results, run_dir, "clustering_results.json")
    
    print("Spectral clustering results:")
    for t_id, lbl in enumerate(cluster_labels):
        t_name = le_task.inverse_transform([t_id])[0]
        print(f" Task={t_name} => cluster {lbl}")
    
    # 9. Reinforcement Learning
    print("\n9. Training RL agent...")
    dummy_resources = [0, 1]  # Example with 2 resources
    env = ProcessEnv(df, le_task, dummy_resources)
    q_table = run_q_learning(env, episodes=30)
    
    # Get optimal policy
    all_actions = [(t, r) for t in env.all_tasks for r in env.resources]
    policy = get_optimal_policy(q_table, all_actions)
    
    # Save RL results
    rl_results = {
        "num_states": len(policy),
        "num_actions": len(all_actions),
        "policy": {
            str(state): {"task": int(action[0]), "resource": int(action[1])}
            for state, action in policy.items()
        }
    }
    save_metrics(rl_results, run_dir, "rl_results.json")
    
    print(f"Learned policy for {len(policy)} states")
    print(f"\nDone! Results saved in {run_dir}")

if __name__ == "__main__":
    main() 