"""
High-performance graph building utilities with sparse matrix operations,
vectorized execution, and minimal memory overhead.
"""

import torch
import torch.sparse as sparse
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import pandas as pd
import time
import logging
import gc
import psutil
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# processmine/data/graph_builder.py (UPDATED FUNCTION)

def build_graph_data(
    df, 
    enhanced: bool = False, 
    batch_size: Optional[int] = None, 
    num_workers: int = 0, 
    verbose: bool = True,
    bidirectional: bool = True,
    limit_nodes: Optional[int] = None,
    mode: str = 'auto'
) -> List[Data]:
    """
    Build graph data with optimized memory usage and vectorized operations
    
    Args:
        df: Process data dataframe
        enhanced: Whether to include edge features
        batch_size: Batch size for memory-efficient processing 
                   (auto-determined if None)
        num_workers: Number of workers for parallel processing
        verbose: Whether to print progress information
        bidirectional: Whether to create bidirectional edges
        limit_nodes: Maximum number of nodes per graph (None for no limit)
        mode: Graph building strategy ('auto', 'standard', 'sparse')
        
    Returns:
        List of graph data objects
    """
    if verbose:
        logger.info(f"Building {'enhanced ' if enhanced else ''}graph data")
    start_time = time.time()
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        logger.warning("No feature columns found. Ensure feature extraction was performed.")
        feature_cols = ["task_id", "resource_id"]  # Use basic features as fallback
    
    # Determine optimal batch size based on available memory if not specified
    if batch_size is None:
        batch_size = _determine_optimal_batch_size(df, feature_cols, enhanced)
        if verbose:
            logger.info(f"Auto-determined batch size: {batch_size} cases")
    
    # Get unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Function for tracking progress
    progress_tracker = _progress_tracker(num_cases, verbose, "Building graphs")
    
    # Statistics for logging
    stats = {"node_counts": [], "edge_counts": []}
    
    # Auto-determine mode if not specified
    if mode == 'auto':
        # Use sparse for larger datasets
        use_sparse = len(df) > 50000 or len(case_ids) > 1000
        mode = 'sparse' if use_sparse else 'standard'
        
        if verbose:
            logger.info(f"Selected graph building mode: {mode}")
    
    # Process in batches with optimized memory usage
    graphs = []
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:batch_end]
        
        # Create local dataframe view - filter with query for efficiency
        batch_df = df.loc[df["case_id"].isin(batch_case_ids)].copy()
        
        # Pre-sort all data to avoid sorting in loop
        batch_df.sort_values(["case_id", "timestamp"], inplace=True)
        
        # Process batch using the specified mode
        if mode == 'sparse':
            batch_graphs = _build_graphs_sparse(batch_df, feature_cols, enhanced, 
                                          bidirectional, limit_nodes, stats)
        else:  # 'standard' mode
            batch_graphs = _build_graphs_standard(batch_df, feature_cols, enhanced, 
                                              bidirectional, limit_nodes, stats)
            
        # Add batch graphs to overall list
        graphs.extend(batch_graphs)
        
        # Update progress
        progress_tracker(batch_end - i)
        
        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Log statistics
    _log_graph_statistics(graphs, stats, start_time, verbose)
    
    return graphs

def _determine_optimal_batch_size(df, feature_cols, enhanced):
    """Determine optimal batch size based on memory availability"""
    # Estimate memory requirements per case
    avg_events_per_case = len(df) / df["case_id"].nunique()
    bytes_per_event = len(feature_cols) * 4  # 4 bytes per float32
    
    if enhanced:
        # Enhanced graphs need more memory for edge features
        bytes_per_event += 8  # Additional memory for edge features
    
    estimated_memory_per_case = avg_events_per_case * bytes_per_event
    
    # Get available memory with safety margin (use 30% of available)
    available_memory = psutil.virtual_memory().available * 0.3
    
    # Calculate batch size, with sensible min/max
    batch_size = max(50, min(5000, int(available_memory / (estimated_memory_per_case * 1.5))))
    
    return batch_size

def _progress_tracker(total, verbose, desc):
    """Create simple progress tracker function"""
    if not verbose:
        return lambda x: None
        
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total, desc=desc)
        return lambda x: progress_bar.update(x)
    except ImportError:
        # Simple fallback
        start_time = time.time()
        processed = 0
        
        def update(x):
            nonlocal processed
            processed += x
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            print(f"\r{desc}: {processed}/{total} ({100*processed/total:.1f}%) "
                  f"[{elapsed:.1f}s elapsed, {remaining:.1f}s remaining]", end="")
            if processed >= total:
                print()  # New line at end
        
        return update

def _build_graphs_standard(batch_df, feature_cols, enhanced, bidirectional, limit_nodes, stats):
    """Build graphs with standard approach for smaller batches"""
    batch_graphs = []
    
    # Process each case
    for case_id, case_df in batch_df.groupby("case_id"):
        # Sort by timestamp for proper sequence
        case_df = case_df.sort_values("timestamp")
        
        # Apply node limit if specified
        if limit_nodes and len(case_df) > limit_nodes:
            case_df = case_df.iloc[:limit_nodes]
        
        # Create node features
        x_data = torch.tensor(case_df[feature_cols].values, dtype=torch.float32)
        
        n_nodes = len(case_df)
        stats["node_counts"].append(n_nodes)
        
        if n_nodes > 1:
            # Create sequential edges efficiently
            src = torch.arange(n_nodes-1, dtype=torch.long)
            tgt = torch.arange(1, n_nodes, dtype=torch.long)
            
            if bidirectional:
                # For bidirectional edges: both src→tgt and tgt→src
                edge_index = torch.stack([
                    torch.cat([src, tgt]),
                    torch.cat([tgt, src])
                ])
                edge_count = 2 * (n_nodes - 1)
            else:
                # For directional edges: only src→tgt
                edge_index = torch.stack([src, tgt])
                edge_count = n_nodes - 1
                
            stats["edge_counts"].append(edge_count)
            
            # Add edge features if enhanced
            if enhanced:
                # Calculate time differences
                timestamps = case_df["timestamp"].values
                time_diffs = np.array([
                    (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                    for i in range(n_nodes-1)
                ], dtype=np.float32)
                
                # Normalize time differences
                max_time = max(time_diffs.max(), 1e-6)  # Avoid division by zero
                norm_time_diffs = time_diffs / max_time
                
                if bidirectional:
                    # Edge features: forward edges have time diff, backward edges have -time diff
                    edge_attr = torch.tensor(
                        np.concatenate([norm_time_diffs, -norm_time_diffs]), 
                        dtype=torch.float32
                    ).view(-1, 1)
                else:
                    edge_attr = torch.tensor(norm_time_diffs, dtype=torch.float32).view(-1, 1)
                
                # Create graph data object with edge attributes
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    y=torch.tensor(case_df["next_task"].values, dtype=torch.long)
                )
            else:
                # Create graph data object without edge attributes
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index, 
                    y=torch.tensor(case_df["next_task"].values, dtype=torch.long)
                )
        else:
            # Handle single-node case (no edges)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            stats["edge_counts"].append(0)
            
            if enhanced:
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr,
                    y=torch.tensor(case_df["next_task"].values, dtype=torch.long)
                )
            else:
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index,
                    y=torch.tensor(case_df["next_task"].values, dtype=torch.long)
                )
        
        batch_graphs.append(data_obj)
    
    return batch_graphs

def _build_graphs_sparse(batch_df, feature_cols, enhanced, bidirectional, limit_nodes, stats):
    """
    Build graphs using sparse matrix operations for larger batches
    This is more memory efficient for large batches
    """
    # Group case IDs
    case_groups = batch_df.groupby("case_id")
    all_graphs = []
    
    # Extract all node features at once
    all_features = batch_df[feature_cols].values
    
    # Track node offsets
    node_offset = 0
    edge_indices_src = []
    edge_indices_dst = []
    edge_attrs = []
    graph_slices = []
    y_values = []
    
    # Process each case
    for case_id, indices in case_groups.indices.items():
        case_df = batch_df.iloc[indices].sort_values("timestamp")
        
        # Apply node limit if specified
        if limit_nodes and len(case_df) > limit_nodes:
            case_df = case_df.iloc[:limit_nodes]
            indices = indices[:limit_nodes]
        
        n_nodes = len(case_df)
        stats["node_counts"].append(n_nodes)
        
        if n_nodes > 1:
            # Create sequential edges efficiently
            src = np.arange(n_nodes-1) + node_offset
            tgt = np.arange(1, n_nodes) + node_offset
            
            if bidirectional:
                edge_indices_src.extend(src)
                edge_indices_src.extend(tgt)
                edge_indices_dst.extend(tgt)
                edge_indices_dst.extend(src)
                edge_count = 2 * (n_nodes - 1)
            else:
                edge_indices_src.extend(src)
                edge_indices_dst.extend(tgt)
                edge_count = n_nodes - 1
                
            stats["edge_counts"].append(edge_count)
            
            # Add edge features if enhanced
            if enhanced:
                # Calculate time differences
                timestamps = case_df["timestamp"].values
                time_diffs = np.array([
                    (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                    for i in range(n_nodes-1)
                ], dtype=np.float32)
                
                # Normalize time differences
                max_time = max(time_diffs.max(), 1e-6)  # Avoid division by zero
                norm_time_diffs = time_diffs / max_time
                
                if bidirectional:
                    # Edge features: forward edges have time diff, backward edges have -time diff
                    edge_attrs.extend(norm_time_diffs)
                    edge_attrs.extend(-norm_time_diffs)
                else:
                    edge_attrs.extend(norm_time_diffs)
        else:
            stats["edge_counts"].append(0)
        
        # Get target values for this case
        y_values.extend(case_df["next_task"].values)
        
        # Store slice boundaries
        graph_slices.append((node_offset, node_offset + n_nodes))
        
        # Update offset
        node_offset += n_nodes
    
    # Create graphs from slices
    for i, (start, end) in enumerate(graph_slices):
        # Extract node features for this graph
        x_data = torch.tensor(all_features[start:end], dtype=torch.float32)
        
        # Find edges for this graph
        if bidirectional:
            edge_mask = ((start <= np.array(edge_indices_src)) & (np.array(edge_indices_src) < end) & 
                        (start <= np.array(edge_indices_dst)) & (np.array(edge_indices_dst) < end))
        else:
            edge_mask = ((start <= np.array(edge_indices_src)) & (np.array(edge_indices_src) < end) & 
                        (start <= np.array(edge_indices_dst)) & (np.array(edge_indices_dst) < end))
        
        # Extract edges for this graph
        if np.any(edge_mask):
            src = np.array(edge_indices_src)[edge_mask] - start
            dst = np.array(edge_indices_dst)[edge_mask] - start
            edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
            
            if enhanced:
                # Extract edge attributes for this graph
                edge_attr = torch.tensor(np.array(edge_attrs)[edge_mask], dtype=torch.float32).view(-1, 1)
                
                # Create graph data object with edge attributes
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr, 
                    y=torch.tensor(y_values[start:end], dtype=torch.long)
                )
            else:
                # Create graph data object without edge attributes
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index, 
                    y=torch.tensor(y_values[start:end], dtype=torch.long)
                )
        else:
            # Handle no edges case
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
            if enhanced:
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr,
                    y=torch.tensor(y_values[start:end], dtype=torch.long)
                )
            else:
                data_obj = Data(
                    x=x_data, 
                    edge_index=edge_index,
                    y=torch.tensor(y_values[start:end], dtype=torch.long)
                )
        
        all_graphs.append(data_obj)
    
    return all_graphs

def _log_graph_statistics(graphs, stats, start_time, verbose):
    """Log detailed statistics about the built graphs"""
    if not verbose:
        return
    
    # Calculate statistics
    if stats["node_counts"] and stats["edge_counts"]:
        avg_nodes = np.mean(stats["node_counts"])
        avg_edges = np.mean(stats["edge_counts"])
        max_nodes = np.max(stats["node_counts"])
        max_edges = np.max(stats["edge_counts"]) if stats["edge_counts"] else 0
        min_nodes = np.min(stats["node_counts"])
        min_edges = np.min(stats["edge_counts"]) if stats["edge_counts"] else 0
        median_nodes = np.median(stats["node_counts"])
        median_edges = np.median(stats["edge_counts"]) if stats["edge_counts"] else 0
        total_nodes = sum(stats["node_counts"])
        total_edges = sum(stats["edge_counts"]) if stats["edge_counts"] else 0
    else:
        avg_nodes = avg_edges = max_nodes = max_edges = 0
        min_nodes = min_edges = median_nodes = median_edges = 0
        total_nodes = total_edges = 0
    
    # Log detailed statistics
    logger.info(f"Graph statistics:")
    logger.info(f"  Total graphs: {len(graphs):,}")
    logger.info(f"  Total nodes: {total_nodes:,}, Total edges: {total_edges:,}")
    logger.info(f"  Avg nodes per graph: {avg_nodes:.2f} (min={min_nodes}, median={median_nodes}, max={max_nodes})")
    logger.info(f"  Avg edges per graph: {avg_edges:.2f} (min={min_edges}, median={median_edges}, max={max_edges})")
    logger.info(f"  Sparsity: {total_edges/(total_nodes**2):.6f}")
    logger.info(f"Graphs built in {time.time() - start_time:.2f}s")
    
    # Check for potential memory issues
    if max_nodes > 1000 or max_edges > 5000:
        logger.warning(f"Very large graphs detected. Consider limiting graph size with limit_nodes parameter.")

def build_heterogeneous_graph(
    df: pd.DataFrame, 
    node_types: Optional[Dict[str, List[str]]] = None, 
    edge_types: Optional[List[str]] = None, 
    batch_size: int = 1000, 
    verbose: bool = True,
    use_edge_attr: bool = True
) -> List[Dict]:
    """
    Build heterogeneous graph data with optimized memory management
    
    Args:
        df: Process data dataframe
        node_types: Dictionary mapping node types to feature columns
                    (e.g., {'task': ['task_id', 'feat_task_id'], 'resource': ['resource_id']})
        edge_types: List of edge types to create
                    (e.g., ['task_to_task', 'task_to_resource', 'resource_to_task'])
        batch_size: Batch size for memory-efficient processing
        verbose: Whether to print progress information
        use_edge_attr: Whether to include edge attributes
        
    Returns:
        List of heterogeneous graph data objects compatible with PyG
    """
    if verbose:
        logger.info("Building heterogeneous graph data")
    start_time = time.time()
    
    # Define default node types if not provided
    if node_types is None:
        task_cols = [col for col in df.columns if col.startswith('feat_task') or col == 'task_id']
        resource_cols = [col for col in df.columns if col.startswith('feat_resource') or col == 'resource_id']
        
        node_types = {
            "task": task_cols,
            "resource": resource_cols
        }
    
    # Define default edge types if not provided
    if edge_types is None:
        edge_types = ["task_to_task", "task_to_resource", "resource_to_task"]
    
    # Get unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Function for tracking progress
    progress_tracker = _progress_tracker(num_cases, verbose, "Building heterogeneous graphs")
    
    # Process in batches
    het_graphs = []
    
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:batch_end]
        
        # Filter dataframe for current batch
        batch_df = df.loc[df["case_id"].isin(batch_case_ids)].copy()
        
        # Pre-sort by case_id and timestamp
        batch_df.sort_values(["case_id", "timestamp"], inplace=True)
        
        # Process each case
        for case_id, case_group in batch_df.groupby("case_id"):
            # Dictionary to store all node data
            node_data = {}
            
            # Dictionary to map node IDs to indices
            node_indices = {}
            
            # Dictionary to store edge data for each edge type
            edge_data = {}
            
            # Extract nodes and features for each node type
            for node_type, feature_cols in node_types.items():
                # Get valid feature columns
                valid_cols = [col for col in feature_cols if col in case_group.columns]
                
                if not valid_cols:
                    if verbose:
                        logger.warning(f"No valid columns found for node type {node_type}")
                    continue
                
                # Get unique node IDs
                if node_type == "task":
                    node_ids = case_group["task_id"].values
                    id_col = "task_id"
                elif node_type == "resource":
                    node_ids = case_group["resource_id"].values
                    id_col = "resource_id"
                else:
                    # For custom node types, use the first column as ID
                    id_col = valid_cols[0]
                    node_ids = case_group[id_col].values
                
                # Extract features
                if len(valid_cols) > 1:
                    # Use all feature columns
                    features = case_group[valid_cols].values
                else:
                    # Use ID column as feature
                    features = case_group[id_col].values.reshape(-1, 1)
                
                # Convert to tensors
                node_features = torch.tensor(features, dtype=torch.float32)
                
                # Store node data
                node_data[node_type] = node_features
                
                # Map node IDs to indices
                for idx, node_id in enumerate(node_ids):
                    if node_type not in node_indices:
                        node_indices[node_type] = {}
                    node_indices[node_type][node_id] = idx
            
            # Create edges for each edge type
            for edge_type in edge_types:
                if edge_type == "task_to_task":
                    # Sequential task transitions
                    n_tasks = len(case_group)
                    if n_tasks > 1:
                        sources = list(range(n_tasks - 1))
                        targets = list(range(1, n_tasks))
                        
                        # Create edge indices
                        edge_indices = torch.tensor([sources, targets], dtype=torch.long)
                        
                        # Create edge attributes if enabled
                        if use_edge_attr:
                            # Add time difference as edge attribute
                            if "timestamp" in case_group.columns:
                                timestamps = case_group["timestamp"].values
                                time_diffs = np.array([
                                    (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours directly in hours using NumPy time units
                                    for i in range(n_tasks-1)
                                ], dtype=np.float32)
                                
                                edge_attrs = torch.tensor(time_diffs, dtype=torch.float32).view(-1, 1)
                            else:
                                # Default edge attributes
                                edge_attrs = torch.ones(len(sources), 1, dtype=torch.float32)
                        else:
                            edge_attrs = None
                        
                        # Store edge data
                        edge_data[edge_type] = {
                            "edge_index": edge_indices,
                            "edge_attr": edge_attrs
                        }
                
                elif edge_type == "task_to_resource":
                    # Connect tasks to resources
                    if "task_id" in case_group.columns and "resource_id" in case_group.columns:
                        task_indices = []
                        resource_indices = []
                        
                        for idx, row in case_group.iterrows():
                            task_id = row["task_id"]
                            resource_id = row["resource_id"]
                            
                            if "task" in node_indices and task_id in node_indices["task"] and \
                               "resource" in node_indices and resource_id in node_indices["resource"]:
                                task_idx = node_indices["task"][task_id]
                                resource_idx = node_indices["resource"][resource_id]
                                
                                task_indices.append(task_idx)
                                resource_indices.append(resource_idx)
                        
                        if task_indices and resource_indices:
                            # Create edge indices
                            edge_indices = torch.tensor([task_indices, resource_indices], dtype=torch.long)
                            
                            # Create edge attributes
                            if use_edge_attr:
                                # Default edge attributes
                                edge_attrs = torch.ones(len(task_indices), 1, dtype=torch.float32)
                            else:
                                edge_attrs = None
                            
                            # Store edge data
                            edge_data[edge_type] = {
                                "edge_index": edge_indices,
                                "edge_attr": edge_attrs
                            }
                
                elif edge_type == "resource_to_task":
                    # Connect resources to tasks (reverse of task_to_resource)
                    if "task_to_resource" in edge_data:
                        task_to_resource = edge_data["task_to_resource"]
                        
                        # Reverse source and target indices
                        edge_indices = torch.stack([
                            task_to_resource["edge_index"][1],
                            task_to_resource["edge_index"][0]
                        ])
                        
                        # Create edge attributes
                        if use_edge_attr and "edge_attr" in task_to_resource and task_to_resource["edge_attr"] is not None:
                            # Use same edge attributes but potentially modify them to indicate reverse direction
                            edge_attrs = task_to_resource["edge_attr"].clone()
                        else:
                            edge_attrs = None
                        
                        # Store edge data
                        edge_data[edge_type] = {
                            "edge_index": edge_indices,
                            "edge_attr": edge_attrs
                        }
                
                # Add support for custom edge types
                elif "_to_" in edge_type:
                    source_type, target_type = edge_type.split("_to_")
                    
                    if source_type in node_indices and target_type in node_indices:
                        # Get mapping between node types
                        # This requires domain knowledge about the relationship
                        # Here we assume a simple approach: each node in source type connects to all nodes in target type
                        # This should be customized based on actual data relationships
                        
                        src_indices = []
                        tgt_indices = []
                        
                        for src_id, src_idx in node_indices[source_type].items():
                            for tgt_id, tgt_idx in node_indices[target_type].items():
                                src_indices.append(src_idx)
                                tgt_indices.append(tgt_idx)
                        
                        if src_indices and tgt_indices:
                            # Create edge indices
                            edge_indices = torch.tensor([src_indices, tgt_indices], dtype=torch.long)
                            
                            # Create edge attributes
                            if use_edge_attr:
                                edge_attrs = torch.ones(len(src_indices), 1, dtype=torch.float32)
                            else:
                                edge_attrs = None
                            
                            # Store edge data
                            edge_data[edge_type] = {
                                "edge_index": edge_indices,
                                "edge_attr": edge_attrs
                            }
            
            # Create heterogeneous graph data object
            het_graph = {
                "num_nodes": {
                    node_type: data.size(0) for node_type, data in node_data.items()
                },
                "node_features": node_data,
                "edge_indices": {
                    edge_type: data["edge_index"] for edge_type, data in edge_data.items()
                },
                "edge_attrs": {
                    edge_type: data["edge_attr"] for edge_type, data in edge_data.items() 
                    if "edge_attr" in data and data["edge_attr"] is not None
                },
                "y": torch.tensor(case_group["next_task"].values, dtype=torch.long) 
                     if "next_task" in case_group.columns else None
            }
            
            # Add metadata
            het_graph["metadata"] = {
                "case_id": case_id,
                "num_events": len(case_group),
                "node_types": list(node_data.keys()),
                "edge_types": list(edge_data.keys())
            }
            
            het_graphs.append(het_graph)
            
        # Update progress
        progress_tracker(batch_end - i)
    
    # Log completion
    if verbose:
        logger.info(f"Built {len(het_graphs)} heterogeneous graphs in {time.time() - start_time:.2f}s")
    
    return het_graphs