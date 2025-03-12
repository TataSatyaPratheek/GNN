"""
Memory-efficient graph building utilities for process mining.
Optimized for handling large datasets with minimal memory footprint.
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

def build_graph_data(df, enhanced=False, batch_size=None, num_workers=0, verbose=True):
    """
    Build graph data with optimized memory usage through batch processing
    
    Args:
        df: Process data dataframe
        enhanced: Whether to include edge features
        batch_size: Optional batch size for memory-efficient processing 
                   (auto-determined if None)
        num_workers: Number of workers for parallel processing
        verbose: Whether to print progress information
        
    Returns:
        List of graph data objects
    """
    if verbose:
        print(f"\n==== Building {'Enhanced ' if enhanced else ''}Graph Data ====")
    start_time = time.time()
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        logger.warning("No feature columns found. Ensure feature extraction was performed.")
        feature_cols = ["task_id", "resource_id"]  # Use basic features as fallback
    
    # Determine optimal batch size based on available memory if not specified
    if batch_size is None:
        # Estimate memory requirements per case
        avg_events_per_case = len(df) / df["case_id"].nunique()
        estimated_memory_per_case = avg_events_per_case * (len(feature_cols) + 5) * 8  # Conservative estimate
        
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
            desc="Building graphs",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
    
    # Track statistics for logging
    node_counts = []
    edge_counts = []
    
    # Process in batches
    graphs = []
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:i+batch_end]
        
        # Filter dataframe for current batch (more memory efficient)
        batch_df = df[df["case_id"].isin(batch_case_ids)].copy()
        
        # Process each case in the batch
        batch_graphs = []
        for cid, cdata in batch_df.groupby("case_id"):
            # Sort by timestamp for proper sequence
            cdata = cdata.sort_values("timestamp")
            
            # Create node features
            x_data = torch.tensor(cdata[feature_cols].values, dtype=torch.float)
            
            # Build edges
            n_nodes = len(cdata)
            node_counts.append(n_nodes)
            
            if n_nodes > 1:
                # Create sequential edges efficiently
                src = torch.arange(n_nodes-1)
                tgt = torch.arange(1, n_nodes)
                
                # For bidirectional edges: both src→tgt and tgt→src
                edge_index = torch.stack([
                    torch.cat([src, tgt]),
                    torch.cat([tgt, src])
                ])
                
                edge_count = 2 * (n_nodes - 1)  # Bidirectional edges
                edge_counts.append(edge_count)
                
                # Add edge features if enhanced
                if enhanced:
                    # Calculate time differences
                    timestamps = cdata["timestamp"].values
                    time_diffs = np.zeros(n_nodes-1, dtype=np.float32)
                    
                    # Vectorized time difference calculation
                    time_diffs = np.array([
                        (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                        for i in range(n_nodes-1)
                    ])
                    
                    # Normalize time differences
                    max_time = max(time_diffs.max(), 1e-6)  # Avoid division by zero
                    norm_time_diffs = time_diffs / max_time
                    
                    # Edge features: forward edges have time diff, backward edges have -time diff
                    edge_attr = torch.tensor(
                        np.concatenate([norm_time_diffs, -norm_time_diffs]), 
                        dtype=torch.float
                    ).view(-1, 1)
                    
                    # Create graph data object with edge attributes
                    data_obj = Data(
                        x=x_data, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=torch.tensor(cdata["next_task"].values, dtype=torch.long)
                    )
                else:
                    # Create graph data object without edge attributes
                    data_obj = Data(
                        x=x_data, 
                        edge_index=edge_index, 
                        y=torch.tensor(cdata["next_task"].values, dtype=torch.long)
                    )
            else:
                # Handle single-node case (no edges)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_counts.append(0)
                
                if enhanced:
                    edge_attr = torch.empty((0, 1), dtype=torch.float)
                    data_obj = Data(
                        x=x_data, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr,
                        y=torch.tensor(cdata["next_task"].values, dtype=torch.long)
                    )
                else:
                    data_obj = Data(
                        x=x_data, 
                        edge_index=edge_index,
                        y=torch.tensor(cdata["next_task"].values, dtype=torch.long)
                    )
            
            batch_graphs.append(data_obj)
        
        # Add batch graphs to overall list
        graphs.extend(batch_graphs)
        
        # Update progress
        if verbose:
            progress_bar.update(batch_end - i)
        
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
        max_edges = np.max(edge_counts) if edge_counts else 0
    else:
        avg_nodes = avg_edges = max_nodes = max_edges = 0
    
    # Log detailed statistics
    if verbose:
        print(f"\033[1mGraph Statistics\033[0m:")
        print(f"  Total graphs: \033[96m{len(graphs):,}\033[0m")
        print(f"  Avg nodes per graph: \033[96m{avg_nodes:.2f}\033[0m")
        print(f"  Avg edges per graph: \033[96m{avg_edges:.2f}\033[0m")
        print(f"  Max nodes in a graph: \033[96m{max_nodes}\033[0m")
        print(f"  Max edges in a graph: \033[96m{max_edges}\033[0m")
        print(f"Graphs built in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return graphs

def build_heterogeneous_graph(df, node_types=None, edge_types=None, batch_size=1000, verbose=True):
    """
    Build heterogeneous graph data with role-based nodes and typed edges
    
    Args:
        df: Process data dataframe
        node_types: Dictionary mapping feature columns to node types
        edge_types: List of edge types to create
        batch_size: Batch size for memory-efficient processing
        verbose: Whether to print progress information
        
    Returns:
        List of heterogeneous graph data objects
    """
    if verbose:
        print(f"\n==== Building Heterogeneous Graph Data ====")
    start_time = time.time()
    
    # Define default node types if not provided
    if node_types is None:
        node_types = {
            "task": ["task_id", "feat_task_id"],
            "resource": ["resource_id", "feat_resource_id"]
        }
    
    # Define default edge types if not provided
    if edge_types is None:
        edge_types = ["task_to_task", "task_to_resource", "resource_to_task"]
    
    # Group by case and process in batches
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Create progress bar if verbose
    if verbose:
        progress_bar = tqdm(
            total=num_cases, 
            desc="Building heterogeneous graphs",
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )
    
    # Process in batches
    het_graphs = []
    
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:i+batch_end]
        
        # Filter dataframe for current batch
        batch_df = df[df["case_id"].isin(batch_case_ids)].copy()
        
        # Process each case
        for cid, cdata in batch_df.groupby("case_id"):
            cdata = cdata.sort_values("timestamp")
            
            # Create node features for each node type
            node_features = {}
            for node_type, feature_cols in node_types.items():
                # Extract relevant features
                valid_cols = [col for col in feature_cols if col in cdata.columns]
                if valid_cols:
                    features = cdata[valid_cols].values
                    node_features[node_type] = torch.tensor(features, dtype=torch.float)
            
            # Create edge indices for each edge type
            edge_indices = {}
            edge_attrs = {}
            
            # Create task-to-task edges (sequential flow)
            if "task_to_task" in edge_types and len(cdata) > 1:
                n_nodes = len(cdata)
                src = torch.arange(n_nodes-1)
                tgt = torch.arange(1, n_nodes)
                edge_indices["task_to_task"] = torch.stack([src, tgt])
                
                # Add time difference as edge attribute
                timestamps = cdata["timestamp"].values
                time_diffs = np.array([
                    (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')
                    for i in range(n_nodes-1)
                ])
                edge_attrs["task_to_task"] = torch.tensor(time_diffs, dtype=torch.float).view(-1, 1)
            
            # Create task-to-resource edges
            if "task_to_resource" in edge_types:
                # Map tasks to resources
                task_idx = torch.arange(len(cdata))
                resource_idx = torch.arange(len(cdata))  # Same length, resource for each task
                edge_indices["task_to_resource"] = torch.stack([task_idx, resource_idx])
            
            # Build heterogeneous graph data object
            # Note: This is a simplified version; a complete implementation would
            # use torch_geometric's HeteroData class
            het_graph = {
                "num_nodes": len(cdata),
                "node_features": node_features,
                "edge_indices": edge_indices,
                "edge_attrs": edge_attrs,
                "y": torch.tensor(cdata["next_task"].values, dtype=torch.long) if "next_task" in cdata.columns else None
            }
            
            het_graphs.append(het_graph)
        
        # Update progress
        if verbose:
            progress_bar.update(batch_end - i)
        
        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Close progress bar if verbose
    if verbose:
        progress_bar.close()
        print(f"Built {len(het_graphs)} heterogeneous graphs in {time.time() - start_time:.2f}s")
    
    return het_graphs