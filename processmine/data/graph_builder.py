#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-efficient graph building utilities for process mining
"""

import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import time
import gc
import logging
import psutil
from typing import List, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def build_graph_data(df, enhanced=False, batch_size=None, verbose=True):
    """
    Build graph data with optional enhancements
    
    Args:
        df: Process data dataframe
        enhanced: Whether to include edge features
        batch_size: Optional batch size for memory-efficient processing
        verbose: Whether to print progress information
        
    Returns:
        List of graph data objects
    """
    if verbose:
        print(f"\n==== Building {'Enhanced ' if enhanced else ''}Graph Data ====")
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    # Extract unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Create progress bar
    if verbose:
        from tqdm import tqdm
        progress_bar = tqdm(total=num_cases, desc="Building graphs")
    
    # Process data into graphs
    graphs = []
    
    for cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        
        # Create node features
        x_data = torch.tensor(cdata[feature_cols].values, dtype=torch.float)
        
        # Build edges
        n_nodes = len(cdata)
        if n_nodes > 1:
            # Sequential edges (src→tgt and tgt→src)
            src = torch.arange(n_nodes-1)
            tgt = torch.arange(1, n_nodes)
            edge_index = torch.stack([
                torch.cat([src, tgt]),
                torch.cat([tgt, src])
            ])
            
            # Add edge features if enhanced
            if enhanced:
                # Calculate time differences between events
                timestamps = cdata["timestamp"].values
                time_diffs = np.diff(timestamps) / np.timedelta64(1, 'h')
                
                # Normalize time differences
                max_time = max(np.max(time_diffs), 1e-6)
                norm_time_diffs = time_diffs / max_time
                
                # Create edge attributes (forward=positive, backward=negative)
                edge_attr = torch.tensor(
                    np.concatenate([norm_time_diffs, -norm_time_diffs]), 
                    dtype=torch.float
                ).view(-1, 1)
                
                data_obj = Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr, 
                               y=torch.tensor(cdata["next_task"].values, dtype=torch.long))
            else:
                data_obj = Data(x=x_data, edge_index=edge_index, 
                               y=torch.tensor(cdata["next_task"].values, dtype=torch.long))
        else:
            # Single node case (no edges)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
            if enhanced:
                edge_attr = torch.empty((0, 1), dtype=torch.float)
                data_obj = Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr,
                               y=torch.tensor(cdata["next_task"].values, dtype=torch.long))
            else:
                data_obj = Data(x=x_data, edge_index=edge_index,
                               y=torch.tensor(cdata["next_task"].values, dtype=torch.long))
        
        graphs.append(data_obj)
        
        # Update progress
        if verbose:
            progress_bar.update(1)
    
    # Close progress bar
    if verbose:
        progress_bar.close()
    
    return graphs
    """
    Build enhanced graph data with additional edge features and memory optimization
    
    Args:
        df: Process data dataframe
        batch_size: Optional batch size for memory-efficient processing
        verbose: Whether to print progress information
        
    Returns:
        List of enhanced graph data objects
    """
    if verbose:
        print("\n==== Building Enhanced Graph Data ====")
    start_time = time.time()
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure feature extraction was performed correctly.")
    
    # Determine optimal batch size based on available memory if not specified
    if batch_size is None:
        # Estimate memory requirements per case (slightly higher for enhanced graphs)
        avg_events_per_case = len(df) / df["case_id"].nunique()
        estimated_memory_per_case = avg_events_per_case * (len(feature_cols) + 2) * 8  # Extra for edge features
        
        # Get available memory
        available_memory = psutil.virtual_memory().available
        
        # Use at most 30% of available memory
        safe_memory = available_memory * 0.3
        
        # Calculate batch size, with minimum of 100, maximum of 10000
        batch_size = max(100, min(10000, int(safe_memory / estimated_memory_per_case)))
        
        if verbose:
            print(f"Auto-determined batch size: {batch_size} cases")
    
    # Get unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Create progress bar if verbose
    if verbose:
        progress_bar = tqdm(
            total=num_cases, 
            desc="Building enhanced graphs",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
    
    # Track statistics
    edge_counts = []
    node_counts = []
    graphs = []
    
    # Process in batches
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:batch_end]
        
        # Filter dataframe for current batch (more memory efficient)
        batch_df = df[df["case_id"].isin(batch_case_ids)].copy()
        
        # Group by case ID
        for cid, cdata in batch_df.groupby("case_id"):
            cdata = cdata.sort_values("timestamp")

            # Create node features
            x_data = torch.tensor(cdata[feature_cols].values, dtype=torch.float)

            # Create edges between activities
            n_nodes = len(cdata)
            node_counts.append(n_nodes)
            
            if n_nodes > 1:
                # Create standard sequential edges more efficiently 
                src = torch.arange(n_nodes-1)
                tgt = torch.arange(1, n_nodes)
                
                # Add edge features: time between events
                timestamps = cdata["timestamp"].values
                time_diffs = np.zeros(n_nodes-1, dtype=np.float32)
                
                # Vectorized time difference calculation
                for i in range(n_nodes-1):
                    time_diff = (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                    time_diffs[i] = time_diff
                
                # Normalize time differences
                max_time = max(time_diffs.max(), 1e-6)  # Avoid division by zero
                norm_time_diffs = time_diffs / max_time
                
                # Add bidirectional edges with time features
                edge_index = torch.stack([
                    torch.cat([src, tgt]),
                    torch.cat([tgt, src])
                ])
                
                # Edge features: forward edges have time diff, backward edges have -time diff
                edge_attr = torch.tensor(
                    np.concatenate([norm_time_diffs, -norm_time_diffs]), 
                    dtype=torch.float
                ).view(-1, 1)
                
                edge_counts.append(2 * (n_nodes - 1))  # Bidirectional edges
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)
                edge_counts.append(0)
                
            y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
            
            # Create enhanced graph data object with edge attributes
            data_obj = Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr, y=y_data)
            graphs.append(data_obj)
        
        # Update progress
        if verbose:
            progress_bar.update(len(batch_case_ids))
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Free CUDA memory if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Close progress bar if verbose
    if verbose:
        progress_bar.close()
    
    # Calculate statistics
    if edge_counts and node_counts:
        avg_nodes = np.mean(node_counts)
        avg_edges = np.mean(edge_counts)
        max_nodes = np.max(node_counts)
    else:
        avg_nodes = avg_edges = max_nodes = 0
    
    if verbose:
        print(f"\033[1mEnhanced Graph Statistics\033[0m:")
        print(f"  Total graphs: \033[96m{len(graphs):,}\033[0m")
        print(f"  Avg nodes per graph: \033[96m{avg_nodes:.2f}\033[0m")
        print(f"  Avg edges per graph: \033[96m{avg_edges:.2f}\033[0m")
        print(f"  Max nodes in a graph: \033[96m{max_nodes}\033[0m")
        print(f"Enhanced graphs built in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return graphs