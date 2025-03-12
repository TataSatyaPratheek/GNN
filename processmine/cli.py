#!/usr/bin/env python3
"""
Streamlined CLI for ProcessMine with better argument handling, memory optimization, 
and unified interface.
"""
import argparse
import logging
import time
import torch
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("processmine")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_arguments():
    """Parse command line arguments with better organization"""
    parser = argparse.ArgumentParser(
        description="ProcessMine: Memory-Efficient Process Mining with GNN, LSTM, and RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("data_path", help="Path to process data CSV file")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Analysis mode
    analysis_parser = subparsers.add_parser("analyze", help="Analyze process data")
    _add_analysis_arguments(analysis_parser)
    
    # Train mode
    train_parser = subparsers.add_parser("train", help="Train a predictive model")
    _add_train_arguments(train_parser)
    
    # Optimize mode
    optimize_parser = subparsers.add_parser("optimize", help="Optimize process with RL")
    _add_optimize_arguments(optimize_parser)
    
    # Full mode (all steps)
    full_parser = subparsers.add_parser("full", help="Run full pipeline (analyze, train, optimize)")
    _add_full_arguments(full_parser)
    
    # Common arguments for all modes
    for mode_parser in [analysis_parser, train_parser, optimize_parser, full_parser]:
        _add_common_arguments(mode_parser)
    
    args = parser.parse_args()
    
    # Set default mode if not specified
    if args.mode is None:
        args.mode = "analyze"
    
    return args

def _add_common_arguments(parser):
    """Add common arguments for all modes"""
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument("--output-dir", help="Custom output directory")
    output_group.add_argument("--viz-format", choices=["static", "interactive", "both"], 
                            default="both", help="Visualization format")
    
    # System arguments
    system_group = parser.add_argument_group('System')
    system_group.add_argument("--seed", type=int, default=42, help="Random seed")
    system_group.add_argument("--device", help="Computing device (cpu, cuda, cuda:0, etc.)")
    system_group.add_argument("--num-workers", type=int, default=0, 
                             help="Number of worker processes for data loading")
    system_group.add_argument("--mem-efficient", action="store_true", 
                            help="Use memory-efficient mode (slower but uses less memory)")
    system_group.add_argument("--cache-dir", help="Directory to cache processed data")
    system_group.add_argument("--debug", action="store_true", help="Enable debug logging")

def _add_analysis_arguments(parser):
    """Add arguments for analysis mode"""
    analysis_group = parser.add_argument_group('Analysis')
    analysis_group.add_argument("--bottleneck-threshold", type=float, default=90.0,
                             help="Percentile threshold for bottleneck detection")
    analysis_group.add_argument("--freq-threshold", type=int, default=5,
                              help="Minimum frequency for significant transitions")
    analysis_group.add_argument("--max-variants", type=int, default=10,
                              help="Maximum number of process variants to analyze")
    analysis_group.add_argument("--skip-conformance", action="store_true",
                              help="Skip conformance checking (faster)")

def _add_train_arguments(parser):
    """Add arguments for train mode"""
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--model", choices=["gnn", "lstm", "enhanced_gnn", "xgboost", "random_forest"], 
                            default="enhanced_gnn", help="Model type to use")
    model_group.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for neural models")
    model_group.add_argument("--num-layers", type=int, default=2, help="Number of layers in neural models")
    model_group.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    
    # GNN specific arguments
    gnn_group = parser.add_argument_group('GNN Options')
    gnn_group.add_argument("--heads", type=int, default=4, help="Number of attention heads for GAT")
    gnn_group.add_argument("--attention-type", choices=["basic", "positional", "diverse", "combined"], 
                         default="combined", help="Type of attention mechanism")
    gnn_group.add_argument("--pos-dim", type=int, default=16, help="Positional encoding dimension")
    gnn_group.add_argument("--diversity-weight", type=float, default=0.1, help="Weight for diversity loss")
    gnn_group.add_argument("--pooling", choices=["mean", "max", "sum", "combined"], 
                          default="mean", help="Graph pooling method")
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, default=20, help="Training epochs")
    train_group.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_group.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay for regularization")
    train_group.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    train_group.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision (CUDA only)")
    train_group.add_argument("--scheduler", choices=["cosine", "step", "plateau", "linear", "constant"],
                           default="cosine", help="Learning rate scheduler")
    train_group.add_argument("--warmup-epochs", type=int, default=3, help="Epochs for learning rate warmup")
    train_group.add_argument("--clip-grad", type=float, help="Gradient clipping norm (None for no clipping)")
    train_group.add_argument("--class-weight-method", choices=["balanced", "log", "sqrt", "none"],
                           default="balanced", help="Method to compute class weights")

def _add_optimize_arguments(parser):
    """Add arguments for optimize mode"""
    rl_group = parser.add_argument_group('Reinforcement Learning')
    rl_group.add_argument("--rl-episodes", type=int, default=30, help="Number of RL episodes")
    rl_group.add_argument("--rl-alpha", type=float, default=0.1, help="RL learning rate")
    rl_group.add_argument("--rl-gamma", type=float, default=0.9, help="RL discount factor")
    rl_group.add_argument("--rl-epsilon", type=float, default=0.1, help="RL exploration rate")

def _add_full_arguments(parser):
    """Add combined arguments for full mode"""
    # Include arguments from all other modes
    _add_analysis_arguments(parser)
    _add_train_arguments(parser)
    _add_optimize_arguments(parser)

def setup_environment(args):
    """Setup environment for the run with better error handling"""
    # Set random seeds for reproducibility
    set_random_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"run_{timestamp}"
    
    # Create directory structure
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "models", "visualizations", "analysis", 
            "policies", "metrics", "logs"
        ]
        
        for subdir in subdirs:
            os.makedirs(output_dir / subdir, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        # Set up file logging
        file_handler = logging.FileHandler(output_dir / "logs" / "processmine.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        sys.exit(1)
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Save arguments for reproducibility
    try:
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save arguments: {e}")
    
    return device, output_dir

def set_random_seed(seed):
    """Set random seeds for all relevant libraries"""
    if seed is not None:
        # Python random
        import random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Set deterministic mode for CUDA (may impact performance)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        
        logger.info(f"Random seed set to {seed}")

def setup_device(device_str=None):
    """Set up computing device with improved error handling"""
    # Determine device
    if device_str is None:
        # Auto-detect best device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # GPU memory info
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU Memory: {gpu_mem:.2f} GB")
            
            # Use tensor cores if available
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                logger.debug("Enabled TF32 on Tensor Cores")
            
            # Benchmark mode for better performance on fixed input sizes
            torch.backends.cudnn.benchmark = True
            
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for computation")
    else:
        # Use specified device
        try:
            device = torch.device(device_str)
            if device.type == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA specified but not available, falling back to CPU")
                device = torch.device("cpu")
            elif device.type == 'mps' and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS specified but not available, falling back to CPU")
                device = torch.device("cpu")
        except:
            logger.warning(f"Unknown device: {device_str}, falling back to CPU")
            device = torch.device("cpu")
    
    return device

def run_analysis(args, device, output_dir):
    """Run process analysis with optimized memory usage"""
    logger.info("Starting process analysis...")
    
    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.process_mining.analysis import (
            analyze_bottlenecks,
            analyze_cycle_times,
            analyze_transition_patterns,
            identify_process_variants,
            analyze_resource_workload
        )
        from processmine.utils.memory import log_memory_usage
        
        # Log initial memory usage
        log_memory_usage()
        
        # Load and preprocess data
        logger.info(f"Loading data from {args.data_path}")
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            args.data_path,
            norm_method='l2',
            cache_dir=args.cache_dir,
            use_dtypes=True,
            memory_limit_gb=8.0 if not args.mem_efficient else 2.0
        )
        
        # Log basic data statistics
        logger.info(f"Data loaded: {len(df):,} events, {df['case_id'].nunique():,} cases, " +
                   f"{df['task_id'].nunique():,} activities, {df['resource_id'].nunique():,} resources")
        
        # Run analysis pipeline
        start_time = time.time()
        
        # Step 1: Bottleneck analysis
        logger.info("Analyzing bottlenecks...")
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
            df,
            freq_threshold=args.freq_threshold,
            percentile_threshold=args.bottleneck_threshold
        )
        
        # Step 2: Cycle time analysis
        logger.info("Analyzing cycle times...")
        case_stats, long_cases, p95 = analyze_cycle_times(df)
        
        # Step 3: Transition pattern analysis
        logger.info("Analyzing transition patterns...")
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df)
        
        # Step 4: Process variant analysis
        logger.info("Identifying process variants...")
        variant_stats, variant_sequences = identify_process_variants(
            df,
            max_variants=args.max_variants
        )
        
        # Step 5: Resource workload analysis
        logger.info("Analyzing resource workload...")
        resource_stats = analyze_resource_workload(df)
        
        # Save analysis results
        logger.info("Saving analysis results...")
        
        # Create visualizations
        logger.info("Creating visualizations...")
        from processmine.visualization.viz import ProcessVisualizer
        
        # Determine visualization format
        use_interactive = args.viz_format in ["interactive", "both"]
        use_static = args.viz_format in ["static", "both"]
        
        viz = ProcessVisualizer(
            output_dir=output_dir / "visualizations", 
            force_static=not use_interactive
        )
        
        # Create cycle time visualization
        viz.cycle_time_distribution(
            case_stats["duration_h"].values,
            filename="cycle_time_distribution" + (".html" if use_interactive else ".png")
        )
        
        # Create bottleneck analysis visualization
        viz.bottleneck_analysis(
            bottleneck_stats,
            significant_bottlenecks,
            task_encoder
        )
        
        # Create process flow visualization
        viz.process_flow(
            bottleneck_stats,
            task_encoder,
            significant_bottlenecks
        )
        
        # Create transition heatmap
        viz.transition_heatmap(transitions, task_encoder)
        
        # Create Sankey diagram (if interactive)
        if use_interactive:
            viz.sankey_diagram(transitions, task_encoder)
        
        # Create resource workload visualization
        viz.resource_workload(resource_stats)
        
        # Create dashboard (if interactive)
        if use_interactive:
            viz.create_dashboard(
                df=df,
                cycle_times=case_stats["duration_h"].values,
                bottleneck_stats=bottleneck_stats,
                significant_bottlenecks=significant_bottlenecks,
                task_encoder=task_encoder
            )
        
        # Save metrics
        logger.info("Saving metrics...")
        metrics = {
            "cases": df["case_id"].nunique(),
            "events": len(df),
            "activities": df["task_id"].nunique(),
            "resources": df["resource_id"].nunique(),
            "variants": len(variant_stats),
            "bottlenecks": len(significant_bottlenecks),
            "perf": {
                "top_bottleneck_wait": significant_bottlenecks["mean_hours"].iloc[0] if len(significant_bottlenecks) > 0 else 0,
                "median_cycle_time": case_stats["duration_h"].median(),
                "p95_cycle_time": p95,
                "resource_gini": resource_stats.attrs.get("gini_coefficient", 0),
                "top_variant_pct": variant_stats["percentage"].iloc[0] if len(variant_stats) > 0 else 0
            }
        }
        
        # Save analysis metrics
        with open(output_dir / "metrics" / "analysis_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))
        
        # Log completion
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f}s")
        
        return {
            "df": df,
            "task_encoder": task_encoder,
            "resource_encoder": resource_encoder,
            "bottleneck_stats": bottleneck_stats,
            "significant_bottlenecks": significant_bottlenecks,
            "case_stats": case_stats,
            "transitions": transitions,
            "variant_stats": variant_stats,
            "resource_stats": resource_stats,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_training(args, device, output_dir, analysis_results=None):
    """Run model training with optimized memory usage"""
    logger.info("Starting model training...")
    
    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graph_builder import build_graph_data
        from processmine.utils.memory import log_memory_usage
        from processmine.utils.evaluation import compute_class_weights
        
        # Load data if not provided from analysis
        if analysis_results is None or "df" not in analysis_results:
            logger.info(f"Loading data from {args.data_path}")
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                args.data_path,
                norm_method='l2',
                cache_dir=args.cache_dir,
                use_dtypes=True
            )
        else:
            df = analysis_results["df"]
            task_encoder = analysis_results["task_encoder"]
            resource_encoder = analysis_results["resource_encoder"]
        
        # Log memory usage
        log_memory_usage()
        
        # Build appropriate model
        logger.info(f"Creating {args.model} model...")
        
        if args.model == "gnn" or args.model == "enhanced_gnn":
            # Build graph data
            logger.info("Building graph data...")
            # Set enhanced to True for enhanced_gnn
            use_enhanced = args.model == "enhanced_gnn"
            
            # Use memory-efficient batch processing
            batch_size = 500 if not args.mem_efficient else 100
            
            graphs = build_graph_data(
                df,
                enhanced=use_enhanced,
                batch_size=batch_size,
                num_workers=args.num_workers,
                verbose=True
            )
            
            # Import optimized GNN model
            from processmine.models.gnn.architectures import OptimizedGNN
            
            # Create model
            model = OptimizedGNN(
                input_dim=len([col for col in df.columns if col.startswith("feat_")]),
                hidden_dim=args.hidden_dim,
                output_dim=len(task_encoder.classes_),
                num_layers=args.num_layers,
                heads=args.heads,
                dropout=args.dropout,
                attention_type=args.attention_type,
                pos_enc_dim=args.pos_dim,
                diversity_weight=args.diversity_weight,
                pooling=args.pooling,
                predict_time=False,
                use_batch_norm=True,
                use_residual=True,
                mem_efficient=args.mem_efficient
            )
            
            # Create data loaders
            from torch_geometric.loader import DataLoader
            import numpy as np
            
            # Split indices
            indices = np.arange(len(graphs))
            np.random.shuffle(indices)
            train_idx = indices[:int(0.7 * len(indices))]
            val_idx = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]
            test_idx = indices[int(0.85 * len(indices)):]
            
            # Create data loaders with memory efficiency in mind
            num_workers = args.num_workers if not args.mem_efficient else 0
            pin_memory = args.device != "cpu" and not args.mem_efficient
            
            train_loader = DataLoader(
                [graphs[i] for i in train_idx],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            val_loader = DataLoader(
                [graphs[i] for i in val_idx],
                batch_size=args.batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            test_loader = DataLoader(
                [graphs[i] for i in test_idx],
                batch_size=args.batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            logger.info(f"Created data loaders with {len(train_idx)} train, " +
                       f"{len(val_idx)} validation, {len(test_idx)} test samples")
            
        elif args.model == "lstm":
            # Create sequence dataset
            from processmine.data.loader import create_sequence_dataset
            
            logger.info("Creating sequence dataset...")
            sequences, targets, seq_lengths = create_sequence_dataset(
                df,
                max_seq_len=50,
                min_seq_len=2
            )
            
            # Create LSTM model
            from processmine.models.sequence.lstm import NextActivityLSTM
            
            model = NextActivityLSTM(
                num_cls=len(task_encoder.classes_),
                emb_dim=args.hidden_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
            
            # TODO: Create sequence data loaders
            
        elif args.model in ["xgboost", "random_forest"]:
            # Use traditional ML models
            logger.info("Creating ML model...")
            
            # Prepare feature data
            feature_cols = [col for col in df.columns if col.startswith("feat_")]
            X = df[feature_cols].values
            y = df["next_task"].values
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=args.seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed)
            
            if args.model == "xgboost":
                import xgboost as xgb
                
                # Create model with optimized parameters
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=args.lr,
                    objective="multi:softmax",
                    num_class=len(task_encoder.classes_),
                    n_jobs=-1 if not args.mem_efficient else 1,
                    tree_method="gpu_hist" if device.type == "cuda" else "hist",
                    verbosity=1 if args.debug else 0
                )
                
                # Train model
                logger.info("Training XGBoost model...")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=args.debug
                )
                
                # Evaluate model
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                
                logger.info(f"XGBoost model performance: accuracy={accuracy:.4f}, f1={f1:.4f}")
                
                # Save metrics
                metrics = {
                    "model_type": "xgboost",
                    "accuracy": float(accuracy),
                    "f1_weighted": float(f1),
                    "parameters": model.get_params()
                }
                
                # Save model
                model.save_model(str(output_dir / "models" / "xgboost_model.json"))
                
                # Save metrics
                with open(output_dir / "metrics" / "model_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                
                return {
                    "model": model,
                    "metrics": metrics
                }
                
            elif args.model == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                
                # Create model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    n_jobs=-1 if not args.mem_efficient else 1,
                    random_state=args.seed,
                    class_weight="balanced"
                )
                
                # Train model
                logger.info("Training Random Forest model...")
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                
                logger.info(f"Random Forest model performance: accuracy={accuracy:.4f}, f1={f1:.4f}")
                
                # Save metrics
                metrics = {
                    "model_type": "random_forest",
                    "accuracy": float(accuracy),
                    "f1_weighted": float(f1),
                    "parameters": {
                        "n_estimators": model.n_estimators,
                        "max_depth": model.max_depth
                    }
                }
                
                # Save model
                import pickle
                with open(output_dir / "models" / "random_forest_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                # Save metrics
                with open(output_dir / "metrics" / "model_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                
                return {
                    "model": model,
                    "metrics": metrics
                }
        
        # For neural network models (GNN, LSTM), proceed with training
        if args.model in ["gnn", "enhanced_gnn", "lstm"]:
            logger.info(f"Training {args.model} model for {args.epochs} epochs...")
            
            # Import training utilities
            from processmine.core.training import (
                train_model,
                evaluate_model,
                create_optimizer,
                create_lr_scheduler,
                compute_class_weights
            )
            
            # Create class weights for handling imbalance
            class_weights = compute_class_weights(
                df, 
                len(task_encoder.classes_),
                method=args.class_weight_method if args.class_weight_method != "none" else None
            )
            
            # Move weights to device
            class_weights = class_weights.to(device)
            
            # Create loss function with class weights
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
            # Create optimizer
            optimizer = create_optimizer(
                model,
                optimizer_type="adamw",
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            
            # Create scheduler
            lr_scheduler = create_lr_scheduler(
                optimizer,
                scheduler_type=args.scheduler,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                patience=args.patience
            )
            
            # Train model
            model, metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                model_path=output_dir / "models" / f"{args.model}_best.pt",
                use_amp=args.use_amp,
                clip_grad_norm=args.clip_grad,
                lr_scheduler=lr_scheduler,
                memory_efficient=args.mem_efficient,
                track_memory=True
            )
            
            # Evaluate model
            logger.info("Evaluating model on test set...")
            eval_metrics, predictions, true_labels = evaluate_model(
                model,
                test_loader,
                device=device,
                criterion=criterion,
                detailed=True
            )
            
            # Save training history
            with open(output_dir / "metrics" / "training_history.json", "w") as f:
                history = {k: [float(v) for v in vals] for k, vals in metrics.items() if isinstance(vals, list)}
                json.dump(history, f, indent=2)
            
            # Save evaluation metrics
            with open(output_dir / "metrics" / "model_metrics.json", "w") as f:
                json.dump(eval_metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else (int(o) if isinstance(o, (np.integer, int)) else str(o)))
            
            # Create confusion matrix visualization
            if "confusion_matrix" in eval_metrics:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                cm = np.array(eval_metrics["confusion_matrix"])
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(output_dir / "visualizations" / "confusion_matrix.png")
                plt.close()
            
            return {
                "model": model,
                "metrics": eval_metrics,
                "history": metrics
            }
    
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_optimization(args, device, output_dir, analysis_results=None):
    """Run process optimization with RL"""
    logger.info("Starting process optimization...")
    
    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.process_mining.optimization import ProcessEnv, run_q_learning
        
        # Load data if not provided from analysis
        if analysis_results is None or "df" not in analysis_results:
            logger.info(f"Loading data from {args.data_path}")
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                args.data_path,
                norm_method='l2',
                cache_dir=args.cache_dir,
                use_dtypes=True
            )
        else:
            df = analysis_results["df"]
            task_encoder = analysis_results["task_encoder"]
            resource_encoder = analysis_results["resource_encoder"]
        
        # Create environment with resource constraints
        logger.info("Creating process optimization environment...")
        env = ProcessEnv(
            df,
            task_encoder,
            resources=list(range(min(5, df["resource_id"].nunique())))
        )
        
        # Run Q-learning
        logger.info(f"Running Q-learning for {args.rl_episodes} episodes...")
        results = run_q_learning(
            env,
            episodes=args.rl_episodes,
            alpha=args.rl_alpha,
            gamma=args.rl_gamma,
            epsilon=args.rl_epsilon,
            viz_dir=output_dir / "visualizations",
            policy_dir=output_dir / "policies"
        )
        
        return {
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_full_pipeline(args, device, output_dir):
    """Run the full pipeline: analysis, training, and optimization"""
    # Run analysis first
    logger.info("Running full pipeline: analysis + training + optimization")
    
    # Step 1: Analysis
    analysis_results = run_analysis(args, device, output_dir)
    
    if analysis_results is None:
        logger.error("Analysis failed, stopping pipeline")
        return False
    
    # Step 2: Training
    training_results = run_training(args, device, output_dir, analysis_results)
    
    if training_results is None:
        logger.error("Training failed, continuing with optimization")
    
    # Step 3: Optimization
    optimization_results = run_optimization(args, device, output_dir, analysis_results)
    
    if optimization_results is None:
        logger.error("Optimization failed")
    
    # Create final report
    logger.info("Creating final report...")
    
    # TODO: Implement report generation
    
    return True

def main():
    """Main entry point for ProcessMine"""
    # Record start time
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Setup environment
        device, output_dir = setup_environment(args)
        
        # Run appropriate mode
        if args.mode == "analyze":
            run_analysis(args, device, output_dir)
        elif args.mode == "train":
            run_training(args, device, output_dir)
        elif args.mode == "optimize":
            run_optimization(args, device, output_dir)
        elif args.mode == "full":
            run_full_pipeline(args, device, output_dir)
        
        # Log completion time
        total_time = time.time() - start_time
        logger.info(f"Process mining completed in {total_time:.2f}s")
        logger.info(f"Results saved to {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())