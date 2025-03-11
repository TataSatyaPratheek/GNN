# cli.py - Simplified CLI
#!/usr/bin/env python3
"""
Streamlined CLI for ProcessMine
"""
import argparse
import logging
from pathlib import Path
import sys
import time

from processmine.utils.device import setup_device
from processmine.core.experiment import setup_results_dir
from processmine.data.loader import load_and_preprocess_data
from processmine.models import create_model
from processmine.core.runner import run_analysis

def main():
    """Main entry point for ProcessMine"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process Mining with GNN, LSTM, and RL")
    parser.add_argument("data_path", help="Path to process data CSV file")
    parser.add_argument("--model", choices=["gnn", "lstm", "enhanced_gnn"], default="gnn",
                       help="Model type to use")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Set up device
    device = setup_device()
    
    # Set up output directory
    results_dir = setup_results_dir(args.output_dir)
    
    # Process data
    logging.info(f"Processing data from {args.data_path}")
    df, task_encoder, resource_encoder = load_and_preprocess_data(args.data_path)
    
    # Run analysis
    run_analysis(
        df=df,
        model_type=args.model,
        task_encoder=task_encoder,
        resource_encoder=resource_encoder,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=results_dir,
        skip_rl=args.skip_rl
    )
    
    logging.info(f"Results saved to {results_dir}")

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        logging.info(f"Total execution time: {time.time() - start_time:.2f}s")
    except KeyboardInterrupt:
        logging.warning("Process mining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)