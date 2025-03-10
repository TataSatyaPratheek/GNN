#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for enhanced process mining with GNN, LSTM, and RL
Includes improved progress tracking, error handling, and visuals
"""

import os
import sys
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from datetime import datetime
import json
import time
import shutil
import argparse
import warnings
from colorama import init as colorama_init
from termcolor import colored

# Initialize colorama for cross-platform colored terminal output
colorama_init()

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Enhanced Process Mining with GNN, LSTM, and RL')
    parser.add_argument('data_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for GNN training')
    parser.add_argument('--lstm-epochs', type=int, default=5, help='Number of epochs for LSTM training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--norm-features', action='store_true', help='Use L2 normalization for features')
    parser.add_argument('--skip-rl', action='store_true', help='Skip reinforcement learning step')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM modeling step')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
    return parser.parse_args()

# Setup device with enhanced detection
def setup_device():
    print(colored("\n🔍 Detecting optimal device for computation...", "cyan"))
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
        print(colored(f"✅ Using GPU: {device_name}", "green"))
        
        # Print CUDA details
        cuda_version = torch.version.cuda
        print(colored(f"   CUDA Version: {cuda_version}", "green"))
        print(colored(f"   Available GPUs: {torch.cuda.device_count()}", "green"))
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_free = memory_total - memory_reserved
        
        # Display in GB for better readability
        print(colored(f"   GPU Memory: {memory_total/1e9:.2f} GB total, {memory_free/1e9:.2f} GB free", "green"))
        
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(colored("✅ Using Apple Silicon GPU (MPS)", "green"))
    else:
        device = torch.device("cpu")
        print(colored("⚠️ GPU not available. Using CPU for computation.", "yellow"))
        # Print CPU details
        import platform
        print(colored(f"   CPU: {platform.processor()}", "yellow"))
        print(colored(f"   Available cores: {os.cpu_count()}", "yellow"))
    
    return device

# Function to setup results directory
def setup_results_dir(custom_dir=None):
    """Create organized results directory structure with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use absolute path with optional custom directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "results")
    
    if custom_dir:
        if os.path.isabs(custom_dir):
            run_dir = custom_dir
        else:
            run_dir = os.path.join(script_dir, custom_dir)
    else:
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create subdirectories with descriptive names
    subdirs = {
        "models": "Saved model weights and parameters",
        "visualizations": "Generated plots and diagrams",
        "metrics": "Performance metrics and statistics",
        "analysis": "Process mining analysis results",
        "policies": "RL policies and decision rules"
    }
    
    print(colored("\n📂 Creating project directory structure:", "cyan"))
    
    # Create main directory
    if os.path.exists(run_dir):
        print(colored(f"⚠️ Directory {run_dir} already exists", "yellow"))
    else:
        try:
            os.makedirs(run_dir, exist_ok=True)
            print(colored(f"✅ Created main directory: {run_dir}", "green"))
        except Exception as e:
            print(colored(f"❌ Error creating directory {run_dir}: {e}", "red"))
            sys.exit(1)
    
    # Create subdirectories with descriptions in a neat table
    print(colored("\n📁 Creating subdirectories:", "cyan"))
    print(colored("   ┌─────────────────┬─────────────────────────────────────┐", "cyan"))
    print(colored("   │ Directory       │ Description                         │", "cyan"))
    print(colored("   ├─────────────────┼─────────────────────────────────────┤", "cyan"))
    
    for subdir, description in subdirs.items():
        subdir_path = os.path.join(run_dir, subdir)
        try:
            os.makedirs(subdir_path, exist_ok=True)
            status = "✅"
            color = "green"
        except Exception as e:
            status = "❌"
            color = "red"
            print(colored(f"Error creating {subdir_path}: {e}", "red"))
        
        print(colored(f"   │ {status} {subdir.ljust(14)} │ {description.ljust(37)} │", color))
    
    print(colored("   └─────────────────┴─────────────────────────────────────┘", "cyan"))
    
    # Create README file in the run directory
    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Process Mining Results - {timestamp}\n\n")
        f.write("This directory contains results from process mining analysis using GNN, LSTM, and RL techniques.\n\n")
        f.write("## Directory Structure\n\n")
        for subdir, description in subdirs.items():
            f.write(f"- **{subdir}**: {description}\n")
        f.write("\n## Runtime Information\n\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Time**: {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"- **Command**: {' '.join(sys.argv)}\n")
    
    return run_dir

# Function to save metrics
def save_metrics(metrics_dict, run_dir, filename, pretty=True):
    """Save metrics to JSON file with improved formatting"""
    filepath = os.path.join(run_dir, "metrics", filename)
    
    try:
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(metrics_dict, f, indent=4, sort_keys=True)
            else:
                json.dump(metrics_dict, f)
        
        file_size = os.path.getsize(filepath)
        print(colored(f"✅ Saved metrics to {filename} ({file_size/1024:.1f} KB)", "green"))
        
    except Exception as e:
        print(colored(f"❌ Error saving metrics to {filename}: {e}", "red"))

# Function to print section header
def print_section_header(title, width=80):
    """Print a visually appealing section header"""
    print("\n" + "=" * width)
    print(colored(f" 🔍 {title}", "cyan", attrs=["bold"]))
    print("=" * width)

# Main function
def main():
    # Start timing for performance tracking
    total_start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
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
    
    # Import local modules
    try:
        print_section_header("Importing Modules")
        
        # Import data preprocessing modules
        from modules.data_preprocessing import (
            load_and_preprocess_data,
            create_feature_representation,
            build_graph_data,
            compute_class_weights
        )
        
        # Import model modules
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
        
        # Import process mining modules
        from modules.process_mining import (
            analyze_bottlenecks,
            analyze_cycle_times,
            analyze_rare_transitions,
            perform_conformance_checking,
            analyze_transition_patterns,
            spectral_cluster_graph,
            build_task_adjacency
        )
        
        # Import RL modules
        from modules.rl_optimization import (
            ProcessEnv,
            run_q_learning,
            get_optimal_policy
        )
        
        # Import visualization modules
        from visualization.process_viz import (
            plot_confusion_matrix,
            plot_embeddings,
            plot_cycle_time_distribution,
            plot_process_flow,
            plot_transition_heatmap,
            create_sankey_diagram
        )
        
        print(colored("✅ All modules imported successfully", "green"))
        
    except ImportError as e:
        print(colored(f"❌ Error importing modules: {e}", "red"))
        print(colored("\nPlease install required packages:", "yellow"))
        print(colored("pip install -r requirements.txt", "yellow"))
        sys.exit(1)
    
    # 1. Load and preprocess data
    print_section_header("Loading and Preprocessing Data")
    
    # Verify data path exists
    data_path = args.data_path
    if not os.path.exists(data_path):
        print(colored(f"❌ Error: dataset not found at {data_path}", "red"))
        sys.exit(1)
    
    # Load data with progress feedback
    try:
        print(colored(f"📂 Loading data from: {data_path}", "cyan"))
        df = load_and_preprocess_data(data_path)
        
        # Save a copy of raw data to the results directory
        raw_data_sample = df.head(1000)  # Just a sample to avoid large files
        raw_data_sample.to_csv(os.path.join(analysis_dir, "data_sample.csv"), index=False)
        
        # Create feature representation
        print(colored(f"\n🔍 Creating feature representation (normalization={args.norm_features})", "cyan"))
        df, le_task, le_resource = create_feature_representation(df, use_norm_features=args.norm_features)
        
        # Save preprocessing info
        preproc_info = {
            "num_tasks": len(le_task.classes_),
            "num_resources": len(le_resource.classes_),
            "num_cases": df["case_id"].nunique(),
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
            "feature_normalization": "L2" if args.norm_features else "MinMax",
            "task_distribution": df["task_name"].value_counts().to_dict(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_metrics(preproc_info, run_dir, "preprocessing_info.json")
        
        # Save encoded task and resource mappings
        task_mapping = {idx: name for idx, name in enumerate(le_task.classes_)}
        resource_mapping = {idx: name for idx, name in enumerate(le_resource.classes_)}
        
        mappings = {
            "task_mapping": task_mapping,
            "resource_mapping": resource_mapping
        }
        save_metrics(mappings, run_dir, "feature_mappings.json")
        
    except Exception as e:
        print(colored(f"❌ Error during data preprocessing: {e}", "red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 2. Build graph data
    print_section_header("Building Graph Data")
    
    try:
        print(colored("🔄 Converting process data to graph format...", "cyan"))
        graphs = build_graph_data(df)
        
        # Split into train/validation sets
        train_size = int(len(graphs) * 0.8)
        train_graphs = graphs[:train_size]
        val_graphs = graphs[train_size:]
        
        print(colored(f"✅ Created {len(graphs)} graphs", "green"))
        print(colored(f"   Training set: {len(train_graphs)} graphs", "green"))
        print(colored(f"   Validation set: {len(val_graphs)} graphs", "green"))
        
        # Create data loaders with specified batch size
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
        
    except Exception as e:
        print(colored(f"❌ Error building graph data: {e}", "red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 3. Train GAT model
    print_section_header("Training Graph Attention Network (GAT)")
    
    try:
        num_classes = len(le_task.classes_)
        
        # Compute class weights for imbalanced data
        print(colored("📊 Computing class weights for imbalanced data...", "cyan"))
        class_weights = compute_class_weights(df, num_classes).to(device)
        
        # Initialize model
        print(colored("🧠 Initializing GAT model...", "cyan"))
        gat_model = NextTaskGAT(
            input_dim=5,  # Task, resource, amount, day, hour
            hidden_dim=64,
            output_dim=num_classes,
            num_layers=2,
            heads=4,
            dropout=0.5
        ).to(device)
        
        # Initialize criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(gat_model.parameters(), lr=0.0005, weight_decay=5e-4)
        
        # Set model save path
        gat_model_path = os.path.join(models_dir, "best_gnn_model.pth")
        
        # Train model
        print(colored(f"🏋️ Training GAT model for {args.epochs} epochs...", "cyan"))
        gat_model = train_gat_model(
            gat_model, train_loader, val_loader,
            criterion, optimizer, device,
            num_epochs=args.epochs, model_path=gat_model_path,
            viz_dir=viz_dir  # Pass visualization directory
        )
        
        # Save model architecture summary
        try:
            from torchsummary import summary
            model_summary = summary(gat_model, input_size=(5, 100))
            with open(os.path.join(models_dir, "gat_model_summary.txt"), 'w') as f:
                f.write(str(gat_model))
        except ImportError:
            # If torchsummary is not available, save basic model info
            with open(os.path.join(models_dir, "gat_model_summary.txt"), 'w') as f:
                f.write(str(gat_model))
        
    except Exception as e:
        print(colored(f"❌ Error training GAT model: {e}", "red"))
        import traceback
        traceback.print_exc()
        # Continue with other parts even if GAT training fails
    
    # 4. Evaluate GAT model
    print_section_header("Evaluating GAT Model")
    
    try:
        print(colored("🔍 Running model evaluation...", "cyan"))
        y_true, y_pred, y_prob = evaluate_gat_model(gat_model, val_loader, device)
        
        # Create confusion matrix
        confusion_matrix_path = os.path.join(viz_dir, "gat_confusion_matrix.png")
        
        print(colored("📊 Creating confusion matrix visualization...", "cyan"))
        accuracy, f1_score = plot_confusion_matrix(
            y_true, y_pred, le_task.classes_,
            confusion_matrix_path
        )
        
        # Save GAT metrics
        from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report
        
        # Generate classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate Matthews correlation coefficient
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Save metrics
        gat_metrics = {
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "mcc": float(mcc),
            "class_report": class_report,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_metrics(gat_metrics, run_dir, "gat_metrics.json")
        
        # Try to visualize embeddings if possible
        try:
            print(colored("📊 Visualizing task embeddings...", "cyan"))
            
            # Forward pass to get embeddings (just once, not for all data)
            sample_data = next(iter(val_loader)).to(device)
            gat_model.eval()
            
            # Get embeddings before the final classification layer
            def get_embeddings(model, data):
                with torch.no_grad():
                    for conv in model.convs:
                        data.x = conv(data.x, data.edge_index)
                        data.x = torch.nn.functional.elu(data.x)
                    return data.x.cpu().numpy()
            
            # Get embeddings and visualize
            try:
                from visualization.process_viz import UMAP_AVAILABLE
                
                embeddings = get_embeddings(gat_model, sample_data)
                plot_embeddings(
                    embeddings, 
                    labels=sample_data.y.cpu().numpy(),
                    method="tsne", 
                    save_path=os.path.join(viz_dir, "task_embeddings_tsne.png")
                )
                
                if UMAP_AVAILABLE:
                    plot_embeddings(
                        embeddings, 
                        labels=sample_data.y.cpu().numpy(),
                        method="umap", 
                        save_path=os.path.join(viz_dir, "task_embeddings_umap.png")
                    )
            except Exception as e:
                print(colored(f"⚠️ Could not visualize embeddings: {e}", "yellow"))
                
        except Exception as e:
            print(colored(f"⚠️ Embedding visualization failed: {e}", "yellow"))
        
    except Exception as e:
        print(colored(f"❌ Error evaluating GAT model: {e}", "red"))
        import traceback
        traceback.print_exc()
    
    # 5. Train LSTM model (unless skipped)
    if not args.skip_lstm:
        print_section_header("Training LSTM Model")
        
        try:
            print(colored("🔄 Preparing sequence data...", "cyan"))
            train_seq, test_seq = prepare_sequence_data(df)
            
            print(colored("🔄 Creating padded datasets...", "cyan"))
            X_train_pad, X_train_len, y_train_lstm, max_len_train = make_padded_dataset(train_seq, num_classes)
            X_test_pad, X_test_len, y_test_lstm, max_len_test = make_padded_dataset(test_seq, num_classes)
            
            # Print dataset shapes
            print(colored(f"✅ Training sequences: {len(train_seq):,}", "green"))
            print(colored(f"✅ Testing sequences: {len(test_seq):,}", "green"))
            print(colored(f"✅ Max sequence length: {max(max_len_train, max_len_test)}", "green"))
            
            # Initialize LSTM model
            print(colored("🧠 Initializing LSTM model...", "cyan"))
            lstm_model = NextActivityLSTM(
                num_classes, 
                emb_dim=64, 
                hidden_dim=64, 
                num_layers=1
            ).to(device)
            
            # Set model save path
            lstm_model_path = os.path.join(models_dir, "lstm_next_activity.pth")
            
            # Train model
            print(colored(f"🏋️ Training LSTM model for {args.lstm_epochs} epochs...", "cyan"))
            lstm_model = train_lstm_model(
                lstm_model, 
                X_train_pad, X_train_len, y_train_lstm,
                device, 
                batch_size=args.batch_size, 
                epochs=args.lstm_epochs, 
                model_path=lstm_model_path,
                viz_dir=viz_dir  # Pass visualization directory
            )
            
            # Evaluate LSTM model
            print(colored("🔍 Evaluating LSTM model...", "cyan"))
            preds, probs, targets = evaluate_lstm_model(
                lstm_model, 
                X_test_pad, X_test_len, y_test_lstm, 
                args.batch_size, 
                device
            )
            
            # Calculate and save metrics
            from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
            
            lstm_accuracy = accuracy_score(targets, preds)
            lstm_f1 = f1_score(targets, preds, average='weighted')
            lstm_mcc = matthews_corrcoef(targets, preds)
            
            lstm_metrics = {
                "accuracy": float(lstm_accuracy),
                "f1_score": float(lstm_f1),
                "mcc": float(lstm_mcc),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_metrics(lstm_metrics, run_dir, "lstm_metrics.json")
            
            # Create confusion matrix
            print(colored("📊 Creating LSTM confusion matrix...", "cyan"))
            plot_confusion_matrix(
                targets, preds, le_task.classes_,
                os.path.join(viz_dir, "lstm_confusion_matrix.png")
            )
            
        except Exception as e:
            print(colored(f"❌ Error in LSTM modeling: {e}", "red"))
            import traceback
            traceback.print_exc()
    else:
        print(colored("\n⏩ Skipping LSTM modeling (--skip-lstm flag used)", "yellow"))
    
    # 6. Process Mining Analysis
    print_section_header("Performing Process Mining Analysis")
    
    try:
        # Analyze bottlenecks
        print(colored("🔍 Analyzing process bottlenecks...", "cyan"))
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
        
        # Save bottleneck data
        bottleneck_stats.to_csv(os.path.join(analysis_dir, "bottleneck_stats.csv"), index=False)
        significant_bottlenecks.to_csv(os.path.join(analysis_dir, "significant_bottlenecks.csv"), index=False)
        
        # Analyze cycle times
        print(colored("🔍 Analyzing cycle times...", "cyan"))
        case_merged, long_cases, cut95 = analyze_cycle_times(df)
        
        # Save cycle time data
        case_merged.to_csv(os.path.join(analysis_dir, "case_cycle_times.csv"), index=False)
        long_cases.to_csv(os.path.join(analysis_dir, "long_running_cases.csv"), index=False)
        
        # Analyze rare transitions
        print(colored("🔍 Identifying rare transitions...", "cyan"))
        rare_trans = analyze_rare_transitions(bottleneck_stats)
        rare_trans.to_csv(os.path.join(analysis_dir, "rare_transitions.csv"), index=False)
        
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_metrics(process_analysis, run_dir, "process_analysis.json")
        
    except Exception as e:
        print(colored(f"❌ Error in process mining analysis: {e}", "red"))
        import traceback
        traceback.print_exc()
    
    # 7. Create Process Visualizations
    print_section_header("Creating Process Visualizations")
    
    try:
        # Plot cycle time distribution
        print(colored("📊 Creating cycle time distribution...", "cyan"))
        plot_cycle_time_distribution(
            case_merged["duration_h"].values,
            os.path.join(viz_dir, "cycle_time_distribution.png")
        )
        
        # Plot process flow with bottlenecks
        print(colored("📊 Creating process flow visualization...", "cyan"))
        plot_process_flow(
            bottleneck_stats, le_task, significant_bottlenecks.head(10),
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
            transitions, le_task,
            os.path.join(viz_dir, "transition_probability_heatmap.png")
        )
        
        # Create Sankey diagram
        print(colored("📊 Creating process flow Sankey diagram...", "cyan"))
        create_sankey_diagram(
            transitions, le_task,
            os.path.join(viz_dir, "process_flow_sankey.html")
        )
        
    except Exception as e:
        print(colored(f"❌ Error creating process visualizations: {e}", "red"))
        import traceback
        traceback.print_exc()
    
    # 8. Spectral Clustering
    print_section_header("Performing Spectral Clustering")
    
    try:
        print(colored("🔍 Building task adjacency matrix...", "cyan"))
        adj_matrix = build_task_adjacency(df, num_classes)
        
        print(colored("🔍 Performing spectral clustering...", "cyan"))
        cluster_labels = spectral_cluster_graph(adj_matrix, k=3)
        
        # Save clustering results
        clustering_results = {
            "task_clusters": {
                le_task.inverse_transform([t_id])[0]: int(lbl)
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
        
        # Print clustering results
        print(colored("\n📊 Spectral Clustering Results:", "cyan"))
        for cluster_id in range(np.max(cluster_labels) + 1):
            cluster_tasks = [le_task.inverse_transform([t_id])[0] 
                            for t_id, lbl in enumerate(cluster_labels) if lbl == cluster_id]
            print(colored(f"   Cluster {cluster_id} ({len(cluster_tasks)} tasks):", "green"))
            for task in cluster_tasks[:5]:  # Show only first 5 for readability
                print(colored(f"      - {task}", "green"))
            if len(cluster_tasks) > 5:
                print(colored(f"      - ... and {len(cluster_tasks) - 5} more tasks", "green"))
        
        # Visualize clusters if possible
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            
            # Get embeddings for tasks
            task_features = np.eye(num_classes)  # Simple one-hot encoding
            embeddings = TSNE(n_components=2, random_state=42).fit_transform(task_features)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            for cluster_id in range(np.max(cluster_labels) + 1):
                cluster_idx = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
                plt.scatter(
                    embeddings[cluster_idx, 0], 
                    embeddings[cluster_idx, 1],
                    label=f"Cluster {cluster_id}",
                    alpha=0.7,
                    s=100
                )
            
            # Add labels if not too many tasks
            if num_classes < 30:
                for i, txt in enumerate(le_task.classes_):
                    plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]),
                                fontsize=8, ha='center')
            
            plt.title("Task Clusters Visualization")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "task_clusters.png"))
            plt.close()
            
        except Exception as e:
            print(colored(f"⚠️ Could not visualize clusters: {e}", "yellow"))
        
    except Exception as e:
        print(colored(f"❌ Error in spectral clustering: {e}", "red"))
        import traceback
        traceback.print_exc()
    
    # 9. Reinforcement Learning (unless skipped)
    if not args.skip_rl:
        print_section_header("Training RL Agent")
        
        try:
            print(colored("🔄 Setting up process environment...", "cyan"))
            dummy_resources = [0, 1]  # Example with 2 resources
            env = ProcessEnv(df, le_task, dummy_resources)
            
            print(colored("🏋️ Training reinforcement learning agent...", "cyan"))
            rl_results = run_q_learning(env, episodes=30, viz_dir=viz_dir, policy_dir=policy_dir)
            
            # Extract optimal policy
            print(colored("🔍 Extracting optimal policy...", "cyan"))
            all_actions = [(t, r) for t in env.all_tasks for r in env.resources]
            policy_results = get_optimal_policy(rl_results, all_actions, policy_dir=policy_dir)
            
            # Save policy and results
            save_metrics(policy_results, run_dir, "rl_policy.json")
            
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
            
        except Exception as e:
            print(colored(f"❌ Error in reinforcement learning: {e}", "red"))
            import traceback
            traceback.print_exc()
    else:
        print(colored("\n⏩ Skipping reinforcement learning (--skip-rl flag used)", "yellow"))
    
    # Generate final summary report
    try:
        total_duration = time.time() - total_start_time
        
        # Create summary report
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "total_duration_formatted": f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s",
            "dataset": {
                "filename": os.path.basename(data_path),
                "cases": df["case_id"].nunique(),
                "events": len(df),
                "activities": len(df["task_id"].unique()),
                "resources": len(df["resource_id"].unique())
            },
            "models": {
                "gat": {
                    "accuracy": gat_metrics.get("accuracy", 0) if 'gat_metrics' in locals() else 0
                },
                "lstm": {
                    "accuracy": lstm_metrics.get("accuracy", 0) if 'lstm_metrics' in locals() else 0
                }
            },
            "process_analysis": {
                "bottlenecks": len(significant_bottlenecks) if 'significant_bottlenecks' in locals() else 0,
                "median_cycle_time": float(np.median(case_merged["duration_h"])) if 'case_merged' in locals() else 0,
                "p95_cycle_time": float(cut95) if 'cut95' in locals() else 0
            }
        }
        
        # Save summary
        save_metrics(summary, run_dir, "execution_summary.json")
        
        # Generate summary report in markdown format
        report_path = os.path.join(run_dir, "execution_summary.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Process Mining Execution Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"**Duration:** {summary['total_duration_formatted']}\n\n")
            
            f.write(f"## Dataset Information\n\n")
            f.write(f"- **Filename:** {os.path.basename(data_path)}\n")
            f.write(f"- **Cases:** {summary['dataset']['cases']:,}\n")
            f.write(f"- **Events:** {summary['dataset']['events']:,}\n")
            f.write(f"- **Activities:** {summary['dataset']['activities']}\n")
            f.write(f"- **Resources:** {summary['dataset']['resources']}\n\n")
            
            f.write(f"## Model Performance\n\n")
            f.write(f"- **GAT Model Accuracy:** {summary['models']['gat']['accuracy']:.4f}\n")
            if 'lstm_metrics' in locals():
                f.write(f"- **LSTM Model Accuracy:** {summary['models']['lstm']['accuracy']:.4f}\n\n")
            
            f.write(f"## Process Analysis\n\n")
            f.write(f"- **Significant Bottlenecks:** {summary['process_analysis']['bottlenecks']}\n")
            f.write(f"- **Median Cycle Time:** {summary['process_analysis']['median_cycle_time']:.2f} hours\n")
            f.write(f"- **95th Percentile Cycle Time:** {summary['process_analysis']['p95_cycle_time']:.2f} hours\n\n")
            
            f.write(f"## Generated Artifacts\n\n")
            f.write(f"- **Models:** GAT, LSTM\n")
            f.write(f"- **Visualizations:** Confusion matrices, process flow, bottlenecks, transitions, Sankey diagram\n")
            f.write(f"- **Analysis:** Bottlenecks, cycle times, transitions, clusters\n")
            f.write(f"- **Policies:** RL optimization policies\n")
        
        print(colored(f"\n✅ Execution summary saved to {report_path}", "green"))
        
    except Exception as e:
        print(colored(f"⚠️ Error creating summary report: {e}", "yellow"))
    
    # Print final completion message
    print_section_header("Process Mining Complete")
    print(colored(f"Total Duration: {total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s", "green"))
    print(colored(f"Results saved to: {run_dir}", "green"))
    print(colored(f"To view visualizations, check the '{os.path.join(run_dir, 'visualizations')}' directory", "cyan"))
    print(colored(f"To see detailed metrics, check the '{os.path.join(run_dir, 'metrics')}' directory", "cyan"))
    print(colored(f"To see learned policies, check the '{os.path.join(run_dir, 'policies')}' directory", "cyan"))
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored("\n\n⚠️ Process mining interrupted by user", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n\n❌ Unexpected error: {e}", "red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)