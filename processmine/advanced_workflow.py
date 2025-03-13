#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced ProcessMine example demonstrating all improvements from the enhancement plan.
This example shows:
1. Enhanced Graph Representation with PositionalGATConv 
2. Memory Management Enhancements with MemoryEfficientDataLoader
3. Normalization Reconciliation with adaptive_normalization
4. Enhanced GNN with Expressivity Improvements
5. Multi-Head Attention with Diversity Mechanism
6. Multi-Objective Loss Function
7. Comprehensive Ablation Study
"""

import os
import argparse
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("processmine_example")

def run_example():
    """Run the complete advanced ProcessMine example"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced ProcessMine example")
    parser.add_argument("--data_path", type=str, required=True, help="Path to process data CSV file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="enhanced_gnn", help="Model type")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_ablation", action="store_true", help="Whether to run ablation study")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU if available")
    parser.add_argument("--mem_efficient", action="store_true", help="Whether to use memory-efficient mode")
    
    args = parser.parse_args()
    
    # Set up device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Import ProcessMine modules
    from processmine.core.advanced_workflow import run_advanced_workflow
    from processmine.core.ablation_runner import run_ablation_study
    
    # Run advanced workflow
    logger.info("Running advanced workflow...")
    
    workflow_results = run_advanced_workflow(
        data_path=args.data_path,
        output_dir=output_dir / "advanced_workflow",
        model=args.model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        mem_efficient=args.mem_efficient,
        use_positional_encoding=True,
        use_diverse_attention=True,
        use_multi_objective_loss=True,
        use_adaptive_normalization=True,
        # Additional model configuration
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.5,
        diversity_weight=0.1,
        use_residual=True,
        use_batch_norm=True,
        use_layer_norm=False,
        # Multi-objective loss weights
        task_weight=0.5,
        time_weight=0.3,
        structure_weight=0.2
    )
    
    # Print advanced workflow results
    logger.info("Advanced workflow results:")
    metrics = workflow_results.get("metrics", {})
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    # Plot training history if available
    history = workflow_results.get("history", {})
    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history.get("train_loss", []), label="Train Loss")
        plt.plot(history.get("val_loss", []), label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.savefig(output_dir / "advanced_workflow" / "training_history.png")
    
    # Run ablation study if requested
    if args.run_ablation:
        logger.info("Running ablation study...")
        
        # Define components to test
        ablation_config = {
            "components": [
                "use_positional_encoding",
                "use_diverse_attention",
                "use_batch_norm",
                "use_residual",
                "use_layer_norm"
            ],
            "disable": True,  # Test by disabling each component
            "include_combinations": False  # Don't test combinations (would be too many)
        }
        
        # Run ablation study
        ablation_results = run_ablation_study(
            data_path=args.data_path,
            base_model=args.model,
            output_dir=output_dir / "ablation_study",
            device=device,
            epochs=args.epochs // 2,  # Use fewer epochs for ablation
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            mem_efficient=args.mem_efficient,
            ablation_config=ablation_config
        )
        
        # Print ablation study results
        logger.info("Ablation study results:")
        results = ablation_results.get("results", {})
        for exp_name, exp_results in results.items():
            if "test_acc" in exp_results:
                logger.info(f"  {exp_name}: Test Accuracy = {exp_results['test_acc']:.4f}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    run_example()