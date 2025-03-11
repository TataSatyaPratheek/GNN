#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data processing pipeline for ProcessMine CLI
"""

import os
import time
from termcolor import colored
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def run_data_pipeline(args, run_dir):
    """
    Run the data loading and preprocessing pipeline
    
    Args:
        args: Command line arguments
        run_dir: Results directory
        
    Returns:
        Dictionary with dataframe, graphs, and encoders
    """
    start_time = time.time()
    
    # Check if we're using Phase 1 models and features
    using_phase1 = args.model_type in ['decision_tree', 'random_forest', 'xgboost', 'mlp', 
                                      'positional_gat', 'diverse_gat', 'enhanced_gnn'] or \
                  args.adaptive_norm or args.enhanced_features or args.enhanced_graphs
    
    # Process data based on whether we're using Phase 1 enhancements
    if using_phase1:
        # Load and preprocess data with Phase 1 enhancements
        from processmine.data.loader import load_and_preprocess_data_phase1
        df, graphs, task_encoder, resource_encoder = load_and_preprocess_data_phase1(args.data_path, args)
    else:
        # Use original data preprocessing
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.preprocessing import create_feature_representation
        from processmine.data.graph_builder import build_graph_data
        
        # Verify data path exists
        data_path = args.data_path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        
        # Load data with progress feedback
        print(colored(f"📂 Loading data from: {data_path}", "cyan"))
        df = load_and_preprocess_data(data_path)
        
        # Save a copy of raw data to the results directory
        analysis_dir = os.path.join(run_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        raw_data_sample = df.head(1000)  # Just a sample to avoid large files
        raw_data_sample.to_csv(os.path.join(analysis_dir, "data_sample.csv"), index=False)
        
        # Create feature representation
        print(colored(f"\n🔍 Creating feature representation (normalization={args.norm_features})", "cyan"))
        df, task_encoder, resource_encoder = create_feature_representation(df, use_norm_features=args.norm_features)
        
        # Save preprocessing info
        from processmine.core.experiment import save_metrics
        from datetime import datetime
        
        preproc_info = {
            "num_tasks": len(task_encoder.classes_),
            "num_resources": len(resource_encoder.classes_),
            "num_cases": df["case_id"].nunique(),
            "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
            "feature_normalization": "L2" if args.norm_features else "MinMax",
            "task_distribution": df["task_name"].value_counts().to_dict(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_metrics(preproc_info, run_dir, "preprocessing_info.json")
        
        # Save encoded task and resource mappings
        task_mapping = {idx: name for idx, name in enumerate(task_encoder.classes_)}
        resource_mapping = {idx: name for idx, name in enumerate(resource_encoder.classes_)}
        
        mappings = {
            "task_mapping": task_mapping,
            "resource_mapping": resource_mapping
        }
        save_metrics(mappings, run_dir, "feature_mappings.json")
        
        # Build graph data
        print(colored("🔄 Converting process data to graph format...", "cyan"))
        graphs = build_graph_data(df)
    
    logger.info(f"Data pipeline completed in {time.time() - start_time:.2f}s")
    
    return {
        'df': df,
        'graphs': graphs,
        'task_encoder': task_encoder,
        'resource_encoder': resource_encoder
    }