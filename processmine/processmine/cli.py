#!/usr/bin/env python3
"""
Streamlined CLI for ProcessMine with consolidated configuration-based argument handling
and optimized memory usage. Now using DGL for graph operations.
"""
import argparse
import logging
import time
from dgl.data.utils import load_graphs, save_graphs
import torch
import os
import sys
import json
import numpy as np
import dgl
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import importlib

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

# Default configurations for different model types
DEFAULT_CONFIGS = {
    'gnn': {
    'hidden_dim': 64,
    'num_layers': 2,
    'heads': 4,
    'dropout': 0.5,
    'attention_type': 'basic',
    'pos_dim': 16,
    'diversity_weight': 0.1,
    'pooling': 'mean',
    'predict_time': False,
    'use_batch_norm': True,
    'use_residual': True,
    'use_layer_norm': False,
    # Added DGL-specific parameters
    'sparse_attention': False,
    'use_checkpointing': False,
    'use_edge_features': True,
    'node_embedding_dim': 32,
    'dgl_sampling': 'neighbor'  # Options: 'neighbor', 'topk', 'random'
    },
    'enhanced_gnn': {
        'hidden_dim': 64,
        'num_layers': 2,
        'heads': 4,
        'dropout': 0.5,
        'attention_type': 'combined',
        'pos_dim': 16,
        'diversity_weight': 0.1,
        'pooling': 'mean',
        'predict_time': False,
        'use_batch_norm': True,
        'use_residual': True,
        'use_layer_norm': False,
        # Added DGL-specific parameters
        'sparse_attention': False,
        'use_checkpointing': False,
        'use_edge_features': True,
        'node_embedding_dim': 32,
        'dgl_sampling': 'neighbor'  # Options: 'neighbor', 'topk', 'random'
    },
    'lstm': {
        'hidden_dim': 64,
        'emb_dim': 64,
        'num_layers': 1,
        'dropout': 0.3,
        'bidirectional': False,
        'use_attention': True,
        'use_layer_norm': True
    },
    'enhanced_lstm': {
        'hidden_dim': 64,
        'emb_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'use_gru': False,
        'use_transformer': True,
        'num_heads': 4,
        'use_time_features': True
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'criterion': 'gini',
        'class_weight': 'balanced',
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'multi:softmax'
    }
}

# Configuration schema - defines all CLI arguments in a single structure
CLI_CONFIG = {
    # Common arguments for all modes
    'common': {
        # Output arguments
        'output_dir': {
            'help': "Custom output directory",
            'type': str
        },
        'viz_format': {
            'help': "Visualization format",
            'choices': ["static", "interactive", "both"],
            'default': "both"
        },
        
        # System arguments
        'seed': {
            'help': "Random seed",
            'type': int,
            'default': 42
        },
        'device': {
            'help': "Computing device (cpu, cuda, cuda:0, etc.)",
            'type': str
        },
        'num_workers': {
            'help': "Number of worker processes for data loading",
            'type': int,
            'default': 0
        },
        'mem_efficient': {
            'help': "Use memory-efficient mode (slower but uses less memory)",
            'action': "store_true"
        },
        'cache_dir': {
            'help': "Directory to cache processed data",
            'type': str
        },
        'debug': {
            'help': "Enable debug logging",
            'action': "store_true"
        }
    },
    
    # Mode-specific arguments
    'modes': {
        'analyze': {
            'description': "Train a predictive model",
            'arguments': {
                'bottleneck_threshold': {
                    'help': "Percentile threshold for bottleneck detection",
                    'type': float,
                    'default': 90.0
                },
                'freq_threshold': {
                    'help': "Minimum frequency for significant transitions",
                    'type': int,
                    'default': 5
                },
                'max_variants': {
                    'help': "Maximum number of process variants to analyze",
                    'type': int,
                    'default': 10
                },
                'skip_conformance': {
                    'help': "Skip conformance checking (faster)",
                    'action': "store_true"
                },
                'sparse_attention': {
                'help': "Use sparse attention for large graphs (DGL optimization)",
                'action': "store_true",
                'applies_to': ["gnn", "enhanced_gnn"]
                },
                'use_checkpointing': {
                    'help': "Use gradient checkpointing to save memory (DGL optimization)",
                    'action': "store_true",
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'dgl_sampling': {
                    'help': "Graph sampling method for large graphs",
                    'choices': ["neighbor", "random", "topk", "none"],
                    'default': "neighbor",
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'use_edge_features': {
                    'help': "Whether to use edge features in graph models",
                    'action': "store_true", 
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'node_embedding_dim': {
                    'help': "Node embedding dimension for graph models",
                    'type': int,
                    'default': 32,
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'save_graphs': {
                    'help': "Save processed DGL graphs to file for reuse",
                    'action': "store_true"
                },
                'graphs_path': {
                    'help': "Path to load/save processed DGL graphs",
                    'type': str
                }
            }
        },
        
        'train': {
            'description': "Train a predictive model",
            'arguments': {
                # Model arguments
                'model': {
                    'help': "Model type to use",
                    'choices': ["gnn", "lstm", "enhanced_gnn", "enhanced_lstm", "xgboost", "random_forest"],
                    'default': "enhanced_gnn"
                },
                'hidden_dim': {
                    'help': "Hidden dimension for neural models",
                    'type': int,
                    'default': 64
                },
                'num_layers': {
                    'help': "Number of layers in neural models",
                    'type': int,
                    'default': 2
                },
                'dropout': {
                    'help': "Dropout probability",
                    'type': float,
                    'default': 0.5
                },
                
                # GNN specific arguments
                'heads': {
                    'help': "Number of attention heads for GAT",
                    'type': int,
                    'default': 4,
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'attention_type': {
                    'help': "Type of attention mechanism",
                    'choices': ["basic", "positional", "diverse", "combined"],
                    'default': "combined",
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'pos_dim': {
                    'help': "Positional encoding dimension",
                    'type': int,
                    'default': 16,
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'diversity_weight': {
                    'help': "Weight for diversity loss",
                    'type': float,
                    'default': 0.1,
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                'pooling': {
                    'help': "Graph pooling method",
                    'choices': ["mean", "max", "sum", "combined"],
                    'default': "mean",
                    'applies_to': ["gnn", "enhanced_gnn"]
                },
                
                # Training arguments
                'epochs': {
                    'help': "Training epochs",
                    'type': int,
                    'default': 20
                },
                'batch_size': {
                    'help': "Batch size",
                    'type': int,
                    'default': 32
                },
                'lr': {
                    'help': "Learning rate",
                    'type': float,
                    'default': 0.001
                },
                'weight_decay': {
                    'help': "Weight decay for regularization",
                    'type': float,
                    'default': 5e-4
                },
                'patience': {
                    'help': "Early stopping patience",
                    'type': int,
                    'default': 5
                },
                'use_amp': {
                    'help': "Use automatic mixed precision (CUDA only)",
                    'action': "store_true"
                },
                'scheduler': {
                    'help': "Learning rate scheduler",
                    'choices': ["cosine", "step", "plateau", "linear", "constant"],
                    'default': "cosine"
                },
                'warmup_epochs': {
                    'help': "Epochs for learning rate warmup",
                    'type': int,
                    'default': 3
                },
                'clip_grad': {
                    'help': "Gradient clipping norm (None for no clipping)",
                    'type': float
                },
                'class_weight_method': {
                    'help': "Method to compute class weights",
                    'choices': ["balanced", "log", "sqrt", "none"],
                    'default': "balanced"
                }
            }
        },
        
        'optimize': {
            'description': "Optimize process with RL",
            'arguments': {
                'rl_episodes': {
                    'help': "Number of RL episodes",
                    'type': int,
                    'default': 30
                },
                'rl_alpha': {
                    'help': "RL learning rate",
                    'type': float,
                    'default': 0.1
                },
                'rl_gamma': {
                    'help': "RL discount factor",
                    'type': float,
                    'default': 0.9
                },
                'rl_epsilon': {
                    'help': "RL exploration rate",
                    'type': float,
                    'default': 0.1
                }
            }
        },
        
        'full': {
            'description': "Run full pipeline (analyze, train, optimize)",
            'include_modes': ['analyze', 'train', 'optimize']
        }
    }
}

def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser from CLI configuration
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="ProcessMine: Memory-Efficient Process Mining with DGL, LSTM, and RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("data_path", help="Path to process data CSV file")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Add mode-specific parsers
    for mode, mode_config in CLI_CONFIG['modes'].items():
        mode_parser = subparsers.add_parser(
            mode, 
            help=mode_config.get('description', f"{mode.capitalize()} mode")
        )
        
        # Add mode-specific arguments
        if 'arguments' in mode_config:
            for arg_name, arg_config in mode_config['arguments'].items():
                # Skip arguments that don't apply to the selected model
                if 'applies_to' in arg_config and '--model' in sys.argv:
                    model_idx = sys.argv.index('--model') + 1
                    if model_idx < len(sys.argv) and sys.argv[model_idx] not in arg_config['applies_to']:
                        continue
                
                # Extract kwargs for add_argument
                kwargs = {k: v for k, v in arg_config.items() if k not in ['applies_to']}
                mode_parser.add_argument(f"--{arg_name}", **kwargs)
        
        # Include arguments from other modes if specified
        if 'include_modes' in mode_config:
            for included_mode in mode_config['include_modes']:
                if included_mode in CLI_CONFIG['modes'] and 'arguments' in CLI_CONFIG['modes'][included_mode]:
                    for arg_name, arg_config in CLI_CONFIG['modes'][included_mode]['arguments'].items():
                        # Skip if argument already added
                        if any(a.dest == arg_name for a in mode_parser._actions):
                            continue
                            
                        # Extract kwargs for add_argument
                        kwargs = {k: v for k, v in arg_config.items() if k not in ['applies_to']}
                        mode_parser.add_argument(f"--{arg_name}", **kwargs)
        
        # Add common arguments to each mode parser
        for arg_name, arg_config in CLI_CONFIG['common'].items():
            mode_parser.add_argument(f"--{arg_name}", **arg_config)
    
    return parser

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments namespace
    """
    parser = build_parser()
    args = parser.parse_args()
    
    # Set default mode if not specified
    if args.mode is None:
        args.mode = "analyze"
    
    # Apply model-specific defaults if a model is selected
    if hasattr(args, 'model') and args.model in DEFAULT_CONFIGS:
        model_defaults = DEFAULT_CONFIGS[args.model]
        
        # Only set defaults if not explicitly provided by user
        for param, value in model_defaults.items():
            if hasattr(args, param) and getattr(args, param) is None:
                setattr(args, param, value)
    
    return args

def setup_environment(args):
    """Setup environment for the run with optimized error handling"""
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
        subdirs = ["models", "visualizations", "analysis", "policies", "metrics", "logs"]
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
        
        # DGL
        dgl.random.seed(seed)
            
        logger.info(f"Random seed set to {seed}")

def setup_device(device_str=None):
    """Set up computing device with improved error handling"""
    # Determine device
    if device_str is None:
        # Auto-detect best device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
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

def main():
    """Main entry point for ProcessMine"""
    # Record start time
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Setup environment
        device, output_dir = setup_environment(args)
        
        # Import runner functions
        from processmine.core.runner import (
            run_analysis, run_training, run_optimization, run_full_pipeline
        )
        
        # Convert args to dict for passing to run functions
        run_kwargs = vars(args)
        
        # Run appropriate mode
        if args.mode == "analyze":
            run_analysis(
                data_path=args.data_path,
                output_dir=output_dir,
                device=device,
                **{k: v for k, v in run_kwargs.items() if k not in ['data_path', 'mode', 'output_dir']}
            )
        elif args.mode == "train":
            run_training(
                data_path=args.data_path,
                output_dir=output_dir,
                device=device,
                **{k: v for k, v in run_kwargs.items() if k not in ['data_path', 'mode', 'output_dir']}
            )
        elif args.mode == "optimize":
            run_optimization(
                data_path=args.data_path,
                output_dir=output_dir,
                device=device,
                **{k: v for k, v in run_kwargs.items() if k not in ['data_path', 'mode', 'output_dir']}
            )
        elif args.mode == "full":
            run_full_pipeline(
                data_path=args.data_path,
                output_dir=output_dir,
                device=device,
                **{k: v for k, v in run_kwargs.items() if k not in ['data_path', 'mode', 'output_dir']}
            )
        
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
    
    
"""# Updates to be added to processmine/cli.py

# Add this to the CLI_CONFIG dictionary in the 'modes' section:

'ablation': {
    'description': "Run ablation study to evaluate model components",
    'arguments': {
        'model': {
            'help': "Model type to use as baseline",
            'choices': ["gnn", "lstm", "enhanced_gnn", "enhanced_lstm", "xgboost", "random_forest"],
            'default': "enhanced_gnn"
        },
        'study_name': {
            'help': "Name for ablation study",
            'type': str,
            'default': "ablation_study"
        },
        'n_seeds': {
            'help': "Number of random seeds to use",
            'type': int,
            'default': 3
        },
        'ablate_components': {
            'help': "Components to ablate (comma-separated, e.g. 'use_residual,use_batch_norm')",
            'type': str
        },
        'grid_search': {
            'help': "Run grid search over parameters (JSON string)",
            'type': str
        },
        'parallel': {
            'help': "Run experiments in parallel",
            'action': "store_true"
        },
        'metrics': {
            'help': "Metrics to aggregate (comma-separated)",
            'type': str,
            'default': "accuracy,f1_weighted"
        },
        'epochs': {
            'help': "Training epochs",
            'type': int,
            'default': 20
        },
        'batch_size': {
            'help': "Batch size",
            'type': int,
            'default': 32
        },
        'lr': {
            'help': "Learning rate",
            'type': float,
            'default': 0.001
        },
        'include_modes': ["train"]  # Include train mode arguments
    }
},

# Add this to main() function to handle the ablation study mode:

elif args.mode == "ablation":
    # Import ablation study
    from processmine.core.ablation import AblationStudy
    
    # Create base config from CLI args
    base_config = {k: v for k, v in run_kwargs.items() if k not in ['mode', 'ablate_components', 'grid_search', 'parallel', 'n_seeds', 'metrics', 'study_name']}
    
    # Specify output directory
    from pathlib import Path
    output_dir = Path(output_dir) / "ablation_studies"
    
    # Create random seeds
    import random
    random_seeds = [random.randint(1, 10000) for _ in range(args.n_seeds)]
    
    # Initialize ablation study
    study = AblationStudy(
        base_config=base_config,
        output_dir=output_dir,
        experiment_name=args.study_name,
        device=device,
        save_models=True,
        parallel=args.parallel,
        random_seeds=random_seeds
    )
    
    # Add baseline (unchanged) experiment
    study.add_experiment("baseline", {})
    
    # Add component ablation experiments if specified
    if args.ablate_components:
        components = args.ablate_components.split(',')
        study.add_ablation_experiments(components, disable=True)
    
    # Add grid search experiments if specified
    if args.grid_search:
        import json
        try:
            grid = json.loads(args.grid_search)
            study.add_grid_search(grid)
        except json.JSONDecodeError:
            logger.error("Invalid JSON for grid search")
    
    # Define function to run a single experiment
    def run_experiment(config, device):
        # Use training function from runner
        training_results = run_training(
            data_path=args.data_path,
            output_dir=output_dir / "experiments" / config.get("name", "unnamed"),
            device=device,
            **config
        )
        
        # Extract metrics
        if isinstance(training_results, dict) and "metrics" in training_results:
            return training_results["metrics"]
        return training_results
    
    # Parse metrics
    metrics = args.metrics.split(',') if args.metrics else ["accuracy", "f1_weighted"]
    
    # Run ablation study
    results = study.run_experiments(
        run_function=run_experiment,
        aggregate_metrics=metrics
    )
    
    logger.info(f"Ablation study completed. Results saved to {output_dir / args.study_name}")"""