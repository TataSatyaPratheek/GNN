#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for enhanced process mining with GNN, LSTM, and RL
Includes improved progress tracking, error handling, and visuals
With Phase 1 architecture enhancements for improved performance
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
    
    # Add Phase 1 specific arguments
    add_phase1_arguments(parser)
    
    return parser.parse_args()

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
        "policies": "RL policies and decision rules",
        "ablation": "Ablation study results"  # Added for Phase 1
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

# New Phase 1 functions from main_script_updates.py

def setup_phase1_model(args, df, task_encoder, resource_encoder, device):
    """
    Set up Phase 1 model based on command-line arguments
    
    Args:
        args: Command-line arguments
        df: Process data dataframe
        task_encoder: Task label encoder
        resource_encoder: Resource label encoder
        device: Torch device
        
    Returns:
        Configured model
    """
    # Import appropriate modules based on model type
    if args.model_type == 'decision_tree':
        # Import baseline decision tree model
        try:
            from models.baseline_models import DecisionTreeModel
            model = DecisionTreeModel(
                max_depth=10,
                min_samples_split=5,
                criterion='gini',
                class_weight='balanced'
            )
        except ImportError:
            from baseline_models import DecisionTreeModel
            model = DecisionTreeModel(
                max_depth=10,
                min_samples_split=5,
                criterion='gini',
                class_weight='balanced'
            )
    
    elif args.model_type == 'random_forest':
        # Import baseline random forest model
        try:
            from models.baseline_models import RandomForestModel
            model = RandomForestModel(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                criterion='gini',
                class_weight='balanced',
                n_jobs=-1
            )
        except ImportError:
            from baseline_models import RandomForestModel
            model = RandomForestModel(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                criterion='gini',
                class_weight='balanced',
                n_jobs=-1
            )
    
    elif args.model_type == 'xgboost':
        # Import baseline XGBoost model
        try:
            from models.baseline_models import XGBoostModel
            model = XGBoostModel(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softmax'
            )
        except ImportError:
            from baseline_models import XGBoostModel
            model = XGBoostModel(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softmax'
            )
    
    elif args.model_type == 'mlp':
        # Import baseline MLP model
        import torch.nn as nn
        try:
            from models.baseline_models import BasicMLP
            
            num_classes = len(task_encoder.classes_)
            
            # Calculate input dimension
            num_features = len([col for col in df.columns if col.startswith('feat_')])
            
            model = BasicMLP(
                input_dim=num_features,
                hidden_dims=[args.hidden_dim, args.hidden_dim // 2],
                output_dim=num_classes,
                dropout=args.dropout
            ).to(device)
        except ImportError:
            from baseline_models import BasicMLP
            
            num_classes = len(task_encoder.classes_)
            
            # Calculate input dimension
            num_features = len([col for col in df.columns if col.startswith('feat_')])
            
            model = BasicMLP(
                input_dim=num_features,
                hidden_dims=[args.hidden_dim, args.hidden_dim // 2],
                output_dim=num_classes,
                dropout=args.dropout
            ).to(device)
    
    elif args.model_type == 'lstm':
        # Import baseline LSTM model
        try:
            from models.baseline_models import BasicLSTM
            
            num_classes = len(task_encoder.classes_)
            num_resources = len(resource_encoder.classes_)
            
            model = BasicLSTM(
                num_tasks=num_classes,
                num_resources=num_resources,
                embedding_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            ).to(device)
        except ImportError:
            from baseline_models import BasicLSTM
            
            num_classes = len(task_encoder.classes_)
            num_resources = len(resource_encoder.classes_)
            
            model = BasicLSTM(
                num_tasks=num_classes,
                num_resources=num_resources,
                embedding_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            ).to(device)
    
    elif args.model_type == 'basic_gat':
        # Import basic GAT model (existing implementation)
        from models.gat_model import NextTaskGAT
        
        num_classes = len(task_encoder.classes_)
        
        model = NextTaskGAT(
            input_dim=5,  # Basic features
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            heads=args.num_heads,
            dropout=args.dropout
        ).to(device)
    
    elif args.model_type == 'positional_gat':
        # Import positional GAT model (Phase 1 improvement)
        try:
            from models.position_enhanced_gat import PositionalGATModel
            
            num_classes = len(task_encoder.classes_)
            
            model = PositionalGATModel(
                input_dim=5,  # Basic features
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                pos_dim=args.pos_dim,
                num_layers=args.num_layers,
                heads=args.num_heads,
                dropout=args.dropout
            ).to(device)
        except ImportError:
            from position_enhanced_gat import PositionalGATModel
            
            num_classes = len(task_encoder.classes_)
            
            model = PositionalGATModel(
                input_dim=5,  # Basic features
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                pos_dim=args.pos_dim,
                num_layers=args.num_layers,
                heads=args.num_heads,
                dropout=args.dropout
            ).to(device)
    
    elif args.model_type == 'diverse_gat':
        # Import diverse GAT model (Phase 1 improvement)
        try:
            from models.diverse_attention import DiverseGATModel
            
            num_classes = len(task_encoder.classes_)
            
            model = DiverseGATModel(
                input_dim=5,  # Basic features
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                num_layers=args.num_layers,
                heads=args.num_heads,
                dropout=args.dropout,
                diversity_weight=args.diversity_weight
            ).to(device)
        except ImportError:
            from diverse_attention import DiverseGATModel
            
            num_classes = len(task_encoder.classes_)
            
            model = DiverseGATModel(
                input_dim=5,  # Basic features
                hidden_dim=args.hidden_dim,
                output_dim=num_classes,
                num_layers=args.num_layers,
                heads=args.num_heads,
                dropout=args.dropout,
                diversity_weight=args.diversity_weight
            ).to(device)
    
    elif args.model_type == 'enhanced_gnn':
        # Import enhanced GNN model (complete Phase 1 implementation)
        try:
            from models.enhanced_gnn import create_enhanced_gnn
            
            num_classes = len(task_encoder.classes_)
            
            # Calculate input dimension based on enhanced features
            num_features = len([col for col in df.columns if col.startswith('feat_')])
            
            model = create_enhanced_gnn(
                input_dim=num_features,
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.num_heads,
                dropout=args.dropout,
                pos_dim=args.pos_dim,
                diversity_weight=args.diversity_weight,
                predict_time=args.predict_time
            ).to(device)
        except ImportError:
            from enhanced_gnn import create_enhanced_gnn
            
            num_classes = len(task_encoder.classes_)
            
            # Calculate input dimension based on enhanced features
            num_features = len([col for col in df.columns if col.startswith('feat_')])
            
            model = create_enhanced_gnn(
                input_dim=num_features,
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                heads=args.num_heads,
                dropout=args.dropout,
                pos_dim=args.pos_dim,
                diversity_weight=args.diversity_weight,
                predict_time=args.predict_time
            ).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model

def setup_optimizer_and_loss(model, args, class_weights=None, device=None):
    """
    Set up optimizer and loss function for the model
    
    Args:
        model: The model
        args: Command-line arguments
        class_weights: Optional class weights tensor
        device: Torch device
        
    Returns:
        Tuple of (optimizer, criterion)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Set up optimizer based on model type
    if args.model_type in ['mlp', 'lstm', 'basic_gat', 'positional_gat', 'diverse_gat', 'enhanced_gnn']:
        # For neural models
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr if hasattr(args, 'lr') else 0.001,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 5e-4
        )
        
        # Set up criterion
        if args.model_type == 'enhanced_gnn' and args.predict_time:
            # Multi-objective loss for dual task
            try:
                from utils.losses import ProcessLoss
                criterion = ProcessLoss(
                    task_weight=0.7,
                    time_weight=0.3,
                    class_weights=class_weights
                ).to(device) if device else ProcessLoss(
                    task_weight=0.7,
                    time_weight=0.3,
                    class_weights=class_weights
                )
            except ImportError:
                from multi_objective_loss import ProcessLoss
                criterion = ProcessLoss(
                    task_weight=0.7,
                    time_weight=0.3,
                    class_weights=class_weights
                ).to(device) if device else ProcessLoss(
                    task_weight=0.7,
                    time_weight=0.3,
                    class_weights=class_weights
                )
        else:
            # Standard cross-entropy for classification
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
    else:
        # For non-neural models (scikit-learn based models)
        optimizer = None
        criterion = None
    
    return optimizer, criterion

def load_and_preprocess_data_phase1(data_path, args):
    """
    Load and preprocess data with Phase 1 enhancements
    
    Args:
        data_path: Path to data file
        args: Command-line arguments
        
    Returns:
        Tuple of (df, graphs, task_encoder, resource_encoder)
    """
    from modules.data_preprocessing import load_and_preprocess_data
    
    print_section_header("Loading and Preprocessing Data with Phase 1 Enhancements")
    
    # Load and preprocess data
    df, graphs, task_encoder, resource_encoder = load_and_preprocess_data(
        data_path,
        use_adaptive_norm=args.adaptive_norm,
        enhanced_features=args.enhanced_features,
        enhanced_graphs=args.enhanced_graphs,
        batch_size=args.batch_size
    )
    
    return df, graphs, task_encoder, resource_encoder

def train_and_evaluate_model_phase1(model, graphs, args, criterion, optimizer, device, run_dir):
    """
    Train and evaluate model with Phase 1 enhancements
    
    Args:
        model: The model to train
        graphs: List of graph data objects
        args: Command-line arguments
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device
        run_dir: Run directory path
        
    Returns:
        Trained model and evaluation metrics
    """
    from torch_geometric.loader import DataLoader
    import torch
    import numpy as np
    import time
    
    # Import metrics tracking utilities
    try:
        from utils.ablation import MetricsTracker
    except ImportError:
        try:
            from ablation_utils import MetricsTracker
        except ImportError:
            print(colored("Metrics tracking utilities not found. Will continue without detailed tracking.", "yellow"))
            MetricsTracker = None
    
    print_section_header("Training and Evaluating Model with Phase 1 Enhancements")
    
    # Split data into train/val/test
    from sklearn.model_selection import train_test_split
    
    train_val_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_val_graphs, test_size=0.25, random_state=42)
    
    print(colored(f"Training set: {len(train_graphs)} graphs", "green"))
    print(colored(f"Validation set: {len(val_graphs)} graphs", "green"))
    print(colored(f"Testing set: {len(test_graphs)} graphs", "green"))
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    
    # Setup metrics tracker if available
    metrics_tracker = None
    if MetricsTracker is not None:
        metrics_tracker = MetricsTracker(run_dir, f"{args.model_type}_training")
        metrics_tracker.register_model(
            "model",
            args.model_type,
            {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "pos_dim": args.pos_dim if hasattr(args, 'pos_dim') else None,
                "diversity_weight": args.diversity_weight if hasattr(args, 'diversity_weight') else None,
                "predict_time": args.predict_time if hasattr(args, 'predict_time') else False
            },
            f"{args.model_type} model with Phase 1 enhancements"
        )
        metrics_tracker.start_training()
    
    # Training setup
    epochs = args.epochs
    best_val_loss = float('inf')
    best_model_path = os.path.join(run_dir, 'models', 'best_model.pth')
    patience = 5
    patience_counter = 0
    
    # Use mixed precision for faster training if available
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_amp = scaler is not None
    
    # Check if model is torch.nn.Module (neural model)
    is_neural = isinstance(model, torch.nn.Module)
    
    if is_neural:
        # Train neural model
        print(colored(f"Training {args.model_type} model for {epochs} epochs...", "cyan"))
        
        for epoch in range(1, epochs+1):
            # Training
            model.train()
            train_loss = 0.0
            train_diversity_loss = 0.0
            start_time = time.time()
            
            for batch_data in train_loader:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                
                # Handle different model output formats
                if args.model_type == 'enhanced_gnn':
                    # Enhanced GNN with dictionary output
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(batch_data)
                            
                            if args.predict_time:
                                loss, loss_dict = criterion(
                                    outputs["task_pred"], 
                                    batch_data.y,
                                    time_pred=outputs.get("time_pred"), 
                                    time_target=None  # No time target in synthetic data
                                )
                            else:
                                loss = criterion(outputs["task_pred"], batch_data.y)
                                
                            # Add diversity loss if available
                            if "diversity_loss" in outputs and hasattr(args, 'diversity_weight') and args.diversity_weight > 0:
                                diversity_loss = outputs["diversity_loss"]
                                train_diversity_loss += diversity_loss.item()
                                loss = loss + diversity_loss
                        
                        # Scale gradients
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(batch_data)
                        
                        if args.predict_time:
                            loss, loss_dict = criterion(
                                outputs["task_pred"], 
                                batch_data.y,
                                time_pred=outputs.get("time_pred"), 
                                time_target=None  # No time target in synthetic data
                            )
                        else:
                            loss = criterion(outputs["task_pred"], batch_data.y)
                            
                        # Add diversity loss if available
                        if "diversity_loss" in outputs and hasattr(args, 'diversity_weight') and args.diversity_weight > 0:
                            diversity_loss = outputs["diversity_loss"]
                            train_diversity_loss += diversity_loss.item()
                            loss = loss + diversity_loss
                        
                        loss.backward()
                        optimizer.step()
                
                elif args.model_type in ['diverse_gat']:
                    # DiverseGAT with tuple output (logits, diversity_loss)
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits, diversity_loss = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                            loss = criterion(logits, batch_data.y)
                            
                            if hasattr(args, 'diversity_weight') and args.diversity_weight > 0:
                                train_diversity_loss += diversity_loss.item()
                                loss = loss + diversity_loss
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits, diversity_loss = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        loss = criterion(logits, batch_data.y)
                        
                        if hasattr(args, 'diversity_weight') and args.diversity_weight > 0:
                            train_diversity_loss += diversity_loss.item()
                            loss = loss + diversity_loss
                        
                        loss.backward()
                        optimizer.step()
                
                else:
                    # Standard model with logits output
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                            loss = criterion(logits, batch_data.y)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        loss = criterion(logits, batch_data.y)
                        
                        loss.backward()
                        optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_diversity_loss = train_diversity_loss / len(train_loader) if train_diversity_loss > 0 else 0
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = batch_data.to(device)
                    
                    # Handle different model output formats
                    if args.model_type == 'enhanced_gnn':
                        outputs = model(batch_data)
                        
                        if args.predict_time:
                            loss, _ = criterion(
                                outputs["task_pred"], 
                                batch_data.y,
                                time_pred=outputs.get("time_pred"), 
                                time_target=None
                            )
                            logits = outputs["task_pred"]
                        else:
                            loss = criterion(outputs["task_pred"], batch_data.y)
                            logits = outputs["task_pred"]
                    
                    elif args.model_type in ['diverse_gat']:
                        logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        loss = criterion(logits, batch_data.y)
                    
                    else:
                        logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                        loss = criterion(logits, batch_data.y)
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(logits, 1)
                    total += batch_data.y.size(0)
                    correct += (predicted == batch_data.y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total if total > 0 else 0
            
            # Log epoch metrics
            if metrics_tracker is not None:
                metrics_tracker.log_epoch(epoch, {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "diversity_loss": avg_diversity_loss
                })
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(colored(f"Epoch {epoch}/{epochs} - "
                         f"Train Loss: {avg_train_loss:.4f}, "
                         f"Val Loss: {avg_val_loss:.4f}, "
                         f"Val Acc: {val_accuracy:.4f}, "
                         f"Time: {epoch_time:.2f}s", "green"))
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save(model.state_dict(), best_model_path)
                print(colored(f"  Saved best model (val_loss={best_val_loss:.4f})", "green"))
            else:
                patience_counter += 1
                print(colored(f"  No improvement for {patience_counter}/{patience} epochs", "yellow"))
                
                if patience_counter >= patience:
                    print(colored(f"Early stopping triggered after {epoch} epochs", "yellow"))
                    break
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(best_model_path))
        print(colored(f"Loaded best model from {best_model_path}", "green"))
    
    else:
        # Train sklearn-based model
        print(colored(f"Training {args.model_type} model...", "cyan"))
        
        # Convert graph data to tabular format for sklearn models
        X_train = []
        y_train = []
        
        for graph in train_graphs:
            # Extract node features and labels
            X_train.extend(graph.x.numpy())
            y_train.extend(graph.y.numpy())
        
        # Train the model
        model.fit(np.array(X_train), np.array(y_train))
        
        print(colored(f"Trained {args.model_type} model", "green"))
    
    # End training tracking
    if metrics_tracker is not None:
        metrics_tracker.end_training()
        # Plot learning curves
        metrics_tracker.plot_learning_curves()
    
    # Evaluate on test set
    print_section_header("Evaluating Model on Test Set")
    from evaluate_model_on_test import evaluate_model_on_test
    test_metrics = evaluate_model_on_test(model, test_loader, criterion, device, args, metrics_tracker, run_dir)
    
    # Save metrics
    if metrics_tracker is not None:
        metrics_tracker.save()
    
    # Save test metrics to file
    save_metrics(test_metrics, run_dir, f"{args.model_type}_test_metrics.json")
    
    print(colored(f"Model training and evaluation completed", "green"))
    
    return model, test_metrics

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
    from torch_geometric.loader import DataLoader
    import torch
    
    print_section_header("Running Ablation Study")
    
    # Import ablation utilities
    try:
        from utils.ablation import AblationManager
    except ImportError:
        try:
            from ablation_utils import AblationManager
        except ImportError:
            print(colored(f"Error importing ablation utilities. Make sure utils/ablation.py exists.", "red"))
            return
    
    # Create output directory
    ablation_dir = os.path.join(run_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
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
            from models.enhanced_gnn import create_enhanced_gnn
            
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
                from utils.losses import ProcessLoss
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
                f1_macro = f1_score(y_true, y_pred, average='macro')
                f1_weighted = f1_score(y_true, y_pred, average='weighted')
                precision = precision_score(y_true, y_pred, average='macro')
                recall = recall_score(y_true, y_pred, average='macro')
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

def evaluate_model_on_test(model, test_loader, criterion, device, args, metrics_tracker=None, run_dir=None):
    """
    Evaluate model on test set
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Torch device
        args: Command-line arguments
        metrics_tracker: Optional metrics tracker
        run_dir: Run directory path
        
    Returns:
        Dictionary of evaluation metrics
    """
    import torch
    import numpy as np
    import time
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
    
    # Check if model is torch.nn.Module (neural model)
    is_neural = isinstance(model, torch.nn.Module)
    
    if is_neural:
        # Evaluate neural model
        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                
                # Handle different model output formats
                if args.model_type == 'enhanced_gnn':
                    outputs = model(batch_data)
                    
                    if args.predict_time:
                        loss, _ = criterion(
                            outputs["task_pred"], 
                            batch_data.y,
                            time_pred=outputs.get("time_pred"), 
                            time_target=None
                        )
                        logits = outputs["task_pred"]
                    else:
                        loss = criterion(outputs["task_pred"], batch_data.y)
                        logits = outputs["task_pred"]
                
                elif args.model_type in ['diverse_gat']:
                    logits, _ = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    loss = criterion(logits, batch_data.y)
                
                else:
                    logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)
                    loss = criterion(logits, batch_data.y)
                
                test_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                # Collect true labels, predictions, and probabilities
                y_true.extend(batch_data.y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['test_loss'] = test_loss / len(test_loader)
        
        # Log metrics
        if metrics_tracker is not None:
            metrics_tracker.log_prediction_metrics(y_true, y_pred, 'test')
        
        # Print metrics
        print(colored("\nTest Metrics:", "cyan"))
        for name, value in metrics.items():
            print(colored(f"  {name}: {value:.4f}", "green"))
        
        # Create confusion matrix visualization if run_dir provided
        if run_dir is not None:
            try:
                from sklearn.metrics import confusion_matrix
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Plot confusion matrix
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"{args.model_type.upper()} Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                
                # Save figure
                viz_dir = os.path.join(run_dir, "visualizations")
                cm_path = os.path.join(viz_dir, f"{args.model_type}_confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                
                print(colored(f"Saved confusion matrix to {cm_path}", "green"))
            except Exception as e:
                print(colored(f"Error creating confusion matrix: {e}", "yellow"))
        
        return metrics
    
    else:
        # Evaluate sklearn-based model
        X_test = []
        y_test = []
        
        for graph in test_loader.dataset:
            # Extract node features and labels
            X_test.extend(graph.x.numpy())
            y_test.extend(graph.y.numpy())
        
        # Make predictions
        y_pred = model.predict(np.array(X_test))
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
        
        # Log metrics
        if metrics_tracker is not None:
            metrics_tracker.log_prediction_metrics(y_test, y_pred, 'test')
        
        # Print metrics
        print(colored("\nTest Metrics:", "cyan"))
        for name, value in metrics.items():
            print(colored(f"  {name}: {value:.4f}", "green"))
        
        return metrics

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
    ablation_dir = os.path.join(run_dir, "ablation")  # Added for Phase 1
    
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
    
    # Check if we're using Phase 1 models and features
    using_phase1 = args.model_type in ['decision_tree', 'random_forest', 'xgboost', 'mlp', 
                                       'positional_gat', 'diverse_gat', 'enhanced_gnn'] or \
                  args.adaptive_norm or args.enhanced_features or args.enhanced_graphs
    
    # Process data based on whether we're using Phase 1 enhancements
    if using_phase1:
        # Load and preprocess data with Phase 1 enhancements
        df, graphs, task_encoder, resource_encoder = load_and_preprocess_data_phase1(args.data_path, args)
        
        # Setup Phase 1 model
        model = setup_phase1_model(args, df, task_encoder, resource_encoder, device)
        
        # Compute class weights for imbalanced data
        print(colored("📊 Computing class weights for imbalanced data...", "cyan"))
        num_classes = len(task_encoder.classes_)
        class_weights = compute_class_weights(df, num_classes).to(device)
        
        # Setup optimizer and loss function
        optimizer, criterion = setup_optimizer_and_loss(model, args, class_weights, device)
        
        # Run ablation study if requested
        if args.run_ablation:
            run_ablation_study(args, df, graphs, task_encoder, resource_encoder, run_dir, device)
        
        # Train and evaluate model
        model, metrics = train_and_evaluate_model_phase1(model, graphs, args, criterion, optimizer, device, run_dir)
        
        # Store metrics for later use
        if args.model_type == 'basic_gat':
            gat_metrics = metrics
        elif args.model_type == 'lstm':
            lstm_metrics = metrics
        
    else:
        # Use original data preprocessing
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
            
            # Set task_encoder and resource_encoder for potential use in process mining
            task_encoder = le_task
            resource_encoder = le_resource
            
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
    
    # Process Mining Analysis
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
    
    # Create Process Visualizations
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
        
    except Exception as e:
        print(colored(f"❌ Error creating process visualizations: {e}", "red"))
        import traceback
        traceback.print_exc()
    
    # Spectral Clustering
    print_section_header("Performing Spectral Clustering")
    
    try:
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_metrics(clustering_results, run_dir, "clustering_results.json")
        
        # Print clustering results
        print(colored("\n📊 Spectral Clustering Results:", "cyan"))
        for cluster_id in range(np.max(cluster_labels) + 1):
            cluster_tasks = [task_encoder.inverse_transform([t_id])[0] 
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
                for i, txt in enumerate(task_encoder.classes_):
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
    
    # Reinforcement Learning (unless skipped)
    if not args.skip_rl:
        print_section_header("Training RL Agent")
        
        try:
            print(colored("🔄 Setting up process environment...", "cyan"))
            dummy_resources = [0, 1]  # Example with 2 resources
            env = ProcessEnv(df, task_encoder, dummy_resources)
            
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
                args.model_type: {
                    "accuracy": float(metrics['accuracy']) if using_phase1 and 'metrics' in locals() else 0
                },
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
            if using_phase1 and 'metrics' in locals():
                f.write(f"- **{args.model_type.replace('_', ' ').title()} Accuracy:** {summary['models'][args.model_type]['accuracy']:.4f}\n")
            if 'gat_metrics' in locals():
                f.write(f"- **GAT Model Accuracy:** {summary['models']['gat']['accuracy']:.4f}\n")
            if 'lstm_metrics' in locals():
                f.write(f"- **LSTM Model Accuracy:** {summary['models']['lstm']['accuracy']:.4f}\n\n")
            
            f.write(f"## Process Analysis\n\n")
            f.write(f"- **Significant Bottlenecks:** {summary['process_analysis']['bottlenecks']}\n")
            f.write(f"- **Median Cycle Time:** {summary['process_analysis']['median_cycle_time']:.2f} hours\n")
            f.write(f"- **95th Percentile Cycle Time:** {summary['process_analysis']['p95_cycle_time']:.2f} hours\n\n")
            
            f.write(f"## Generated Artifacts\n\n")
            f.write(f"- **Models:** {args.model_type.replace('_', ' ').title() if using_phase1 else 'GAT, LSTM'}\n")
            f.write(f"- **Visualizations:** Confusion matrices, process flow, bottlenecks, transitions, Sankey diagram\n")
            f.write(f"- **Analysis:** Bottlenecks, cycle times, transitions, clusters\n")
            if not args.skip_rl:
                f.write(f"- **Policies:** RL optimization policies\n")
            if args.run_ablation:
                f.write(f"- **Ablation:** Component contribution analysis\n")
        
        print(colored(f"\n✅ Execution summary saved to {report_path}", "green"))
        
    except Exception as e:
        print(colored(f"⚠️ Error creating summary report: {e}", "yellow"))
    
    # Print final completion message
    print_section_header("Process Mining Complete")
    print(colored(f"Total Duration: {total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s", "green"))
    print(colored(f"Results saved to: {run_dir}", "green"))
    print(colored(f"To view visualizations, check the '{os.path.join(run_dir, 'visualizations')}' directory", "cyan"))
    print(colored(f"To see detailed metrics, check the '{os.path.join(run_dir, 'metrics')}' directory", "cyan"))
    if not args.skip_rl:
        print(colored(f"To see learned policies, check the '{os.path.join(run_dir, 'policies')}' directory", "cyan"))
    if args.run_ablation:
        print(colored(f"To see ablation study results, check the '{os.path.join(run_dir, 'ablation')}' directory", "cyan"))
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