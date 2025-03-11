#!/usr/bin/env python3
"""
Streamlined CLI for ProcessMine
"""
import argparse
import logging
import time
import torch
import os
import sys
from pathlib import Path

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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process Mining with GNN, LSTM, and RL")
    
    # Required arguments
    parser.add_argument("data_path", help="Path to process data CSV file")
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument("--model", choices=["gnn", "lstm", "enhanced_gnn", "xgboost", "random_forest"], 
                            default="gnn", help="Model type to use")
    model_group.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for neural models")
    model_group.add_argument("--num-layers", type=int, default=2, help="Number of layers in neural models")
    model_group.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    
    # GNN specific arguments
    gnn_group = parser.add_argument_group('GNN Options')
    gnn_group.add_argument("--heads", type=int, default=4, help="Number of attention heads for GAT")
    gnn_group.add_argument("--use-positional", action="store_true", help="Use positional encoding in GNN")
    gnn_group.add_argument("--use-diversity", action="store_true", help="Use attention diversity in GNN")
    gnn_group.add_argument("--pos-dim", type=int, default=16, help="Positional encoding dimension")
    gnn_group.add_argument("--diversity-weight", type=float, default=0.1, help="Weight for diversity loss")
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, default=20, help="Training epochs")
    train_group.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_group.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay for regularization")
    train_group.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    train_group.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision (CUDA only)")
    
    # Analysis arguments
    analysis_group = parser.add_argument_group('Analysis')
    analysis_group.add_argument("--skip-analysis", action="store_true", help="Skip process analysis")
    analysis_group.add_argument("--skip-rl", action="store_true", help="Skip RL optimization")
    analysis_group.add_argument("--rl-episodes", type=int, default=30, help="Number of RL episodes")
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument("--output-dir", help="Custom output directory")
    output_group.add_argument("--viz-format", choices=["static", "interactive", "both"], 
                            default="both", help="Visualization format")
    
    # Misc arguments
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument("--seed", type=int, default=42, help="Random seed")
    misc_group.add_argument("--device", help="Computing device (cpu, cuda, cuda:0, etc.)")
    misc_group.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()

def setup_device(device_str=None):
    """Set up computing device with improved error handling"""
    # Determine device
    if device_str is None:
        # Auto-detect best device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Use tensor cores if available
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
                logger.debug("Enabled TF32 on Tensor Cores")
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
        except:
            logger.warning(f"Unknown device: {device_str}, falling back to CPU")
            device = torch.device("cpu")
    
    return device

def setup_environment(args):
    """Setup environment for the run"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    import numpy as np
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"run_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create subdirectories
    os.makedirs(output_dir / "models", exist_ok=True)
    os.makedirs(output_dir / "visualizations", exist_ok=True)
    os.makedirs(output_dir / "analysis", exist_ok=True)
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    return device, output_dir

def main():
    """Main entry point for ProcessMine"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Setup environment
        device, output_dir = setup_environment(args)
        
        # Import core modules
        logger.info("Importing core modules...")
        from processmine.data.loader import load_and_preprocess_data
        from processmine.utils.memory import log_memory_usage
        
        # Log initial memory usage
        log_memory_usage()
        
        # Load and preprocess data
        logger.info(f"Loading data from {args.data_path}")
        df, task_encoder, resource_encoder = load_and_preprocess_data(args.data_path)
        
        # Log basic data statistics
        logger.info(f"Data loaded: {len(df)} events, {df['case_id'].nunique()} cases, " +
                   f"{df['task_id'].nunique()} activities, {df['resource_id'].nunique()} resources")
        
        # Build graph data if using GNN models
        if "gnn" in args.model:
            from processmine.data.graph_builder import build_graph_data
            logger.info("Building graph data...")
            
            # Determine if enhanced graphs should be used based on model type
            use_enhanced = args.model == "enhanced_gnn"
            graphs = build_graph_data(df, enhanced=use_enhanced)
        
        # Create model
        logger.info(f"Creating {args.model} model...")
        if args.model == "gnn":
            from processmine.models.gnn.architectures import ConfigurableGNN
            model = ConfigurableGNN(
                input_dim=len([col for col in df.columns if col.startswith("feat_")]),
                hidden_dim=args.hidden_dim,
                output_dim=len(task_encoder.classes_),
                attention_type="basic",
                heads=args.heads,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.model == "enhanced_gnn":
            from processmine.models.gnn.architectures import ConfigurableGNN
            
            # Determine attention type based on args
            if args.use_positional and args.use_diversity:
                attention_type = "combined"
            elif args.use_positional:
                attention_type = "positional"
            elif args.use_diversity:
                attention_type = "diverse"
            else:
                attention_type = "basic"
            
            model = ConfigurableGNN(
                input_dim=len([col for col in df.columns if col.startswith("feat_")]),
                hidden_dim=args.hidden_dim,
                output_dim=len(task_encoder.classes_),
                attention_type=attention_type,
                heads=args.heads,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pos_dim=args.pos_dim,
                diversity_weight=args.diversity_weight
            )
        elif args.model == "lstm":
            from processmine.models.sequence.lstm import NextActivityLSTM
            model = NextActivityLSTM(
                num_cls=len(task_encoder.classes_),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.model in ["xgboost", "random_forest"]:
            # These are handled differently as they're not PyTorch models
            model = None
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        # Move model to device if it's a PyTorch model
        if model is not None:
            model = model.to(device)
            
            # Log model summary
            from processmine.utils.memory import get_model_size
            model_size = get_model_size(model)
            logger.info(f"Model created with {model_size['param_count']:,} parameters " +
                       f"({model_size['total_mb']:.2f} MB)")
        
        # Split data and create data loaders
        if "gnn" in args.model:
            from torch_geometric.loader import DataLoader
            import numpy as np
            
            # Split indices
            indices = np.arange(len(graphs))
            np.random.shuffle(indices)
            train_idx = indices[:int(0.7 * len(indices))]
            val_idx = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]
            test_idx = indices[int(0.85 * len(indices)):]
            
            # Create data loaders
            train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=args.batch_size)
            test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=args.batch_size)
            
            logger.info(f"Created data loaders with {len(train_idx)} train, " +
                       f"{len(val_idx)} validation, {len(test_idx)} test samples")
            
            # Train model
            logger.info(f"Training model for {args.epochs} epochs...")
            
            # Create optimizer and criterion
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # Use class weights to handle imbalance
            from processmine.utils.evaluation import compute_class_weights
            class_weights = compute_class_weights(df, len(task_encoder.classes_))
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
            
            # Train model
            from processmine.core.training import train_model
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
                use_amp=args.use_amp
            )
            
            # Evaluate model
            logger.info("Evaluating model on test set...")
            from processmine.core.training import evaluate_model
            eval_metrics = evaluate_model(model, test_loader, device)
            
            # Log metrics
            for name, value in eval_metrics.items():
                logger.info(f"  {name}: {value:.4f}")
            
            # Save metrics to file
            import json
            with open(output_dir / "models" / "metrics.json", "w") as f:
                json.dump(eval_metrics, f, indent=2)
        
        # Run process analysis
        if not args.skip_analysis:
            logger.info("Running process analysis...")
            from processmine.process_mining.analysis import (
                analyze_bottlenecks,
                analyze_cycle_times,
                analyze_transition_patterns,
                identify_process_variants,
                analyze_resource_workload
            )
            
            # Run bottleneck analysis
            bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
            
            # Run cycle time analysis
            case_stats, long_cases, p95 = analyze_cycle_times(df)
            
            # Run transition pattern analysis
            transitions, trans_count, prob_matrix = analyze_transition_patterns(df)
            
            # Identify process variants
            variant_stats, variant_sequences = identify_process_variants(df)
            
            # Analyze resource workload
            resource_stats = analyze_resource_workload(df)
            
            # Create visualizations
            logger.info("Creating visualizations...")
            from processmine.visualization.viz import ProcessVisualizer
            
            # Determine visualization format
            use_interactive = args.viz_format in ["interactive", "both"]
            use_static = args.viz_format in ["static", "both"]
            
            viz = ProcessVisualizer(output_dir=output_dir / "visualizations", 
                                   force_static=not use_interactive)
            
            # Create cycle time visualization
            if use_static:
                viz.cycle_time_distribution(case_stats["duration_h"].values)
            
            if use_interactive:
                viz.cycle_time_distribution(case_stats["duration_h"].values, 
                                          filename="cycle_time_interactive.html")
            
            # Create bottleneck analysis visualization
            viz.bottleneck_analysis(bottleneck_stats, significant_bottlenecks, task_encoder)
            
            # Create process flow visualization
            viz.process_flow(bottleneck_stats, task_encoder, significant_bottlenecks)
            
            # Create transition heatmap
            viz.transition_heatmap(transitions, task_encoder)
            
            # Create Sankey diagram (if interactive)
            if use_interactive:
                viz.sankey_diagram(transitions, task_encoder)
            
            # Create resource workload visualization
            viz.resource_workload(resource_stats)
            
            # Create dashboard
            if use_interactive:
                viz.create_dashboard(
                    df=df,
                    cycle_times=case_stats["duration_h"].values,
                    bottleneck_stats=bottleneck_stats,
                    significant_bottlenecks=significant_bottlenecks,
                    task_encoder=task_encoder
                )
        
        # Run RL optimization if requested
        if not args.skip_rl:
            logger.info("Running RL optimization...")
            from processmine.process_mining.optimization import ProcessEnv, run_q_learning
            
            # Create environment
            env = ProcessEnv(df, task_encoder, resources=[0, 1])  # Simplified for example
            
            # Run Q-learning
            results = run_q_learning(
                env, 
                episodes=args.rl_episodes,
                viz_dir=output_dir / "visualizations",
                policy_dir=output_dir / "analysis"
            )
        
        # Create summary report
        logger.info("Creating summary report...")
        from processmine.utils.reporting import generate_report
        report_path = generate_report(
            args=args,
            df=df,
            model_type=args.model,
            metrics=eval_metrics if "eval_metrics" in locals() else None,
            output_dir=output_dir
        )
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Process mining completed in {total_time:.2f}s")
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())