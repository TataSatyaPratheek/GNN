#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading utilities for process mining
"""

import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
import gc
import logging
from colorama import Fore, Style

logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path, use_adaptive_norm=False, enhanced_features=False,
                           enhanced_graphs=False, batch_size=None, required_cols=None):
    """
    Load and preprocess the event log data with progress feedback and XES format support
    
    Args:
        data_path: Path to data file
        use_adaptive_norm: Whether to use adaptive normalization
        enhanced_features: Whether to use enhanced feature engineering 
        enhanced_graphs: Whether to use enhanced graph building
        batch_size: Optional batch size for memory efficiency
        required_cols: List of required columns
        
    Returns:
        Preprocessed dataframe
    """
    print("\n==== Loading and Preprocessing Data ====")
    start_time = time.time()
    
    if required_cols is None:
        required_cols = ["case_id", "task_name", "timestamp", "resource"]
    
    print(f"Loading data from {data_path}...")
    try:
        # Use more efficient CSV reading with chunking for large files
        chunksize = 100000
        chunks = []
        
        # Count total lines for progress bar
        with open(data_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # Read in chunks with progress bar
        with tqdm(total=total_lines, desc="Reading CSV", ncols=100) as pbar:
            for chunk in pd.read_csv(data_path, chunksize=chunksize):
                chunks.append(chunk)
                pbar.update(len(chunk))
                
        df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        print(f"\033[91mError reading CSV in chunks: {e}\033[0m")
        print("Falling back to standard pandas read_csv...")
        df = pd.read_csv(data_path)
    
    print(f"Data loaded: {len(df)} rows, {df.shape[1]} columns")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Rename columns for consistency (supporting XES format)
    df.rename(columns={
        "case:id": "case_id",
        "case:concept:name": "case_name",
        "concept:name": "task_name",
        "org:resource": "resource",
        "time:timestamp": "timestamp",
        "case:Amount": "amount",
        "case:BudgetNumber": "budget_number",
        "case:DeclarationNumber": "declaration_number",
        "org:role": "role"
    }, inplace=True, errors="ignore")

    print(f"Columns after renaming: {df.columns.tolist()}")

    # Validate required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}. Found columns: {df.columns.tolist()}")

    # Process timestamps
    print("Processing timestamps...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    before_na = len(df)
    df.dropna(subset=["timestamp"], inplace=True)
    after_na = len(df)
    if before_na > after_na:
        print(f"\033[93mWarning: Dropped {before_na - after_na} rows with invalid timestamps\033[0m")
    
    # Sort by case ID and timestamp
    print("Sorting data...")
    df.sort_values(["case_id","timestamp"], inplace=True)
    
    # Report basic statistics
    case_count = df["case_id"].nunique()
    task_count = df["task_name"].nunique() 
    resource_count = df["resource"].nunique()
    
    # Calculate time period
    min_date = df["timestamp"].min().strftime('%Y-%m-%d')
    max_date = df["timestamp"].max().strftime('%Y-%m-%d')
    
    print(f"\033[1mData Summary\033[0m:")
    print(f"  Time period: \033[96m{min_date}\033[0m to \033[96m{max_date}\033[0m")
    print(f"  Cases: \033[96m{case_count:,}\033[0m")
    print(f"  Activities: \033[96m{task_count}\033[0m")
    print(f"  Resources: \033[96m{resource_count}\033[0m")
    print(f"  Events: \033[96m{len(df):,}\033[0m")
    
    print(f"Preprocessing completed in \033[96m{time.time() - start_time:.2f}s\033[0m")

    # Additional preprocessing based on flags
    if enhanced_features or use_adaptive_norm:
        # Import the feature representation function
        from processmine.data.preprocessing import create_feature_representation, AdaptiveNormalizer
        
        # Get task and resource encoders
        from sklearn.preprocessing import LabelEncoder
        task_encoder = LabelEncoder().fit(df["task_name"])
        resource_encoder = LabelEncoder().fit(df["resource"])
        
        # Create enhanced feature representation
        print(f"\n🔍 Creating feature representation (adaptive_norm={use_adaptive_norm})")
        df, _, _ = create_feature_representation(df, use_norm_features=use_adaptive_norm)
        
        if enhanced_graphs:
            # Import graph builder
            from processmine.data.graph_builder import build_graph_data
            
            # Build enhanced graph data
            print("\n🔄 Building enhanced graph data...")
            graphs = build_graph_data(df)
            
            return df, graphs, task_encoder, resource_encoder
            
        return df, task_encoder, resource_encoder
        
    return df

def load_and_preprocess_data_phase1(data_path, args):
    """
    Load and preprocess data with Phase 1 enhancements
    
    Args:
        data_path: Path to data file
        args: Command-line arguments
        
    Returns:
        Tuple of (df, graphs, task_encoder, resource_encoder)
    """
    print(f"{Fore.CYAN}🔍 Loading and Preprocessing Data with Phase 1 Enhancements{Style.RESET_ALL}")
    
    # Load and preprocess data
    result = load_and_preprocess_data(
        data_path,
        use_adaptive_norm=args.adaptive_norm,
        enhanced_features=args.enhanced_features,
        enhanced_graphs=args.enhanced_graphs,
        batch_size=args.batch_size
    )
    
    # Ensure we're handling both return types properly
    if isinstance(result, tuple) and len(result) == 4:
        # Already returns a tuple with four elements
        return result
    else:
        # Just returns a dataframe
        df = result
        # Create feature representation
        from processmine.data.preprocessing import create_feature_representation
        from processmine.data.graph_builder import build_graph_data
        df, task_encoder, resource_encoder = create_feature_representation(df, use_norm_features=args.adaptive_norm)
        graphs = build_graph_data(df)
        return df, graphs, task_encoder, resource_encoder