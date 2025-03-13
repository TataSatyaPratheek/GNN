"""
High-performance graph building utilities with DGL, optimized for efficient memory usage
and vectorized operations with minimal memory overhead.
"""

import torch
import numpy as np
import dgl
import pandas as pd
import time
import logging
import gc
import psutil
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def build_graph_data(
    df, 
    enhanced: bool = False, 
    batch_size: Optional[int] = None, 
    num_workers: int = 0, 
    verbose: bool = True,
    bidirectional: bool = True,
    limit_nodes: Optional[int] = None,
    mode: str = 'auto'
) -> List[dgl.DGLGraph]:
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
        List of DGL graph objects
    """
    if verbose:
        logger.info(f"Building {'enhanced ' if enhanced else ''}graph data with DGL")
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
    """Build graphs with standard approach for smaller batches using DGL"""
    batch_graphs = []
    
    # Process each case
    for case_id, case_df in batch_df.groupby("case_id"):
        # Sort by timestamp for proper sequence
        case_df = case_df.sort_values("timestamp")
        
        # Apply node limit if specified
        if limit_nodes and len(case_df) > limit_nodes:
            case_df = case_df.iloc[:limit_nodes]
        
        n_nodes = len(case_df)
        stats["node_counts"].append(n_nodes)
        
        # Create node features
        node_features = torch.tensor(case_df[feature_cols].values, dtype=torch.float32)
        
        # Store targets as a tensor
        labels = torch.tensor(case_df["next_task"].values, dtype=torch.long)
        
        if n_nodes > 1:
            # Create sequential edges efficiently
            src = torch.arange(n_nodes-1, dtype=torch.long)
            dst = torch.arange(1, n_nodes, dtype=torch.long)
            
            # For DGL we use lists of source and destination nodes
            edges_src = src.tolist()
            edges_dst = dst.tolist()
            
            if bidirectional:
                # Add reverse edges
                edges_src.extend(dst.tolist())
                edges_dst.extend(src.tolist())
                edge_count = 2 * (n_nodes - 1)
            else:
                edge_count = n_nodes - 1
                
            stats["edge_counts"].append(edge_count)
            
            # Create DGL graph
            g = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
            
            # Add node features
            g.ndata['feat'] = node_features
            g.ndata['label'] = labels
            
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
                
                # Create edge features tensor
                if bidirectional:
                    # Edge features: forward edges have time diff, backward edges have -time diff
                    edge_feats = torch.tensor(
                        np.concatenate([norm_time_diffs, -norm_time_diffs]), 
                        dtype=torch.float32
                    ).view(-1, 1)
                else:
                    edge_feats = torch.tensor(norm_time_diffs, dtype=torch.float32).view(-1, 1)
                
                # Add edge features to the graph
                g.edata['feat'] = edge_feats
        else:
            # Handle single-node case (no edges)
            g = dgl.graph(([], []), num_nodes=1)
            g.ndata['feat'] = node_features
            g.ndata['label'] = labels
            
            if enhanced:
                # Add empty edge features for consistency
                g.edata['feat'] = torch.empty((0, 1), dtype=torch.float32)
            
            stats["edge_counts"].append(0)
        
        batch_graphs.append(g)
    
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
        node_features = torch.tensor(all_features[start:end], dtype=torch.float32)
        
        # Extract labels for this graph
        labels = torch.tensor(y_values[start:end], dtype=torch.long)
        
        # Find edges for this graph
        if edge_indices_src and edge_indices_dst: # Check if there are any edges
            mask = ((start <= np.array(edge_indices_src)) & (np.array(edge_indices_src) < end) & 
                    (start <= np.array(edge_indices_dst)) & (np.array(edge_indices_dst) < end))
            
            # Extract edges for this graph
            if np.any(mask):
                src = np.array(edge_indices_src)[mask] - start
                dst = np.array(edge_indices_dst)[mask] - start
                
                # Create DGL graph
                g = dgl.graph((src, dst), num_nodes=end-start)
                
                # Add node features
                g.ndata['feat'] = node_features
                g.ndata['label'] = labels
                
                if enhanced and edge_attrs:
                    # Extract edge attributes for this graph
                    edge_feats = torch.tensor(np.array(edge_attrs)[mask], dtype=torch.float32).view(-1, 1)
                    g.edata['feat'] = edge_feats
            else:
                # Create a graph with no edges
                g = dgl.graph(([], []), num_nodes=end-start)
                g.ndata['feat'] = node_features
                g.ndata['label'] = labels
                
                if enhanced:
                    # Add empty edge features for consistency
                    g.edata['feat'] = torch.empty((0, 1), dtype=torch.float32)
        else:
            # Create a graph with no edges
            g = dgl.graph(([], []), num_nodes=end-start)
            g.ndata['feat'] = node_features
            g.ndata['label'] = labels
            
            if enhanced:
                # Add empty edge features for consistency
                g.edata['feat'] = torch.empty((0, 1), dtype=torch.float32)
        
        all_graphs.append(g)
    
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
) -> List[dgl.DGLGraph]:
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
        List of heterogeneous DGL graphs
    """
    if verbose:
        logger.info("Building heterogeneous graph data with DGL")
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
            # Dictionaries to store node features by type
            node_features = {}
            
            # Dictionary to map node IDs to indices
            node_indices = {}
            
            # Dictionaries to store edge information by type
            edge_srcs = {}
            edge_dsts = {}
            edge_features = {}
            
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
                node_features[node_type] = torch.tensor(features, dtype=torch.float32)
                
                # Map node IDs to indices
                for idx, node_id in enumerate(node_ids):
                    if node_type not in node_indices:
                        node_indices[node_type] = {}
                    node_indices[node_type][node_id] = idx
            
            # Process edges for each edge type
            for edge_type in edge_types:
                src_nodes = []
                dst_nodes = []
                attrs = []
                
                if edge_type == "task_to_task":
                    # Sequential task transitions
                    if "task" in node_indices:
                        n_tasks = len(case_group)
                        if n_tasks > 1:
                            # Create sequential edges
                            for i in range(n_tasks - 1):
                                src_nodes.append(i)
                                dst_nodes.append(i + 1)
                            
                            # Add edge features if enabled
                            if use_edge_attr and "timestamp" in case_group.columns:
                                timestamps = case_group["timestamp"].values
                                for i in range(n_tasks - 1):
                                    time_diff = (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')
                                    attrs.append([float(time_diff)])
                
                elif edge_type == "task_to_resource":
                    # Connect tasks to resources
                    if "task" in node_indices and "resource" in node_indices:
                        for idx, row in case_group.iterrows():
                            if hasattr(row, "task_id") and hasattr(row, "resource_id"):
                                task_id = row["task_id"]
                                resource_id = row["resource_id"]
                                
                                if task_id in node_indices["task"] and resource_id in node_indices["resource"]:
                                    src_nodes.append(node_indices["task"][task_id])
                                    dst_nodes.append(node_indices["resource"][resource_id])
                                    
                                    # Add placeholder edge features if enabled
                                    if use_edge_attr:
                                        attrs.append([1.0])  # Default edge attribute
                
                elif edge_type == "resource_to_task":
                    # Just reverse of task_to_resource
                    if "task_to_resource" in edge_srcs and "task_to_resource" in edge_dsts:
                        src_nodes = edge_dsts["task_to_resource"]
                        dst_nodes = edge_srcs["task_to_resource"]
                        
                        # Copy attributes if present
                        if use_edge_attr and "task_to_resource" in edge_features:
                            attrs = edge_features["task_to_resource"]
                
                # Store the edge information if we have edges
                if src_nodes and dst_nodes:
                    edge_srcs[edge_type] = src_nodes
                    edge_dsts[edge_type] = dst_nodes
                    
                    # FIX: Check properly if attrs has elements before converting to tensor
                    if use_edge_attr and len(attrs) > 0:
                        edge_features[edge_type] = torch.tensor(attrs, dtype=torch.float32)
            
            # Create DGL heterogeneous graph
            if node_features and edge_srcs:
                # Check what node types and edge types we actually have data for
                available_ntypes = list(node_features.keys())
                available_etypes = []
                
                # Validate edge types
                for e_type in edge_srcs.keys():
                    if "_to_" in e_type:
                        src_type, dst_type = e_type.split("_to_")
                        if src_type in available_ntypes and dst_type in available_ntypes:
                            available_etypes.append((src_type, e_type, dst_type))
                
                # Create the graph data dict
                graph_data = {}
                for src_type, e_type, dst_type in available_etypes:
                    src = edge_srcs[e_type]
                    dst = edge_dsts[e_type]
                    if src and dst:
                        graph_data[(src_type, e_type, dst_type)] = (src, dst)
                
                # Create heterogeneous graph if we have edges
                if graph_data:
                    g = dgl.heterograph(graph_data)
                    
                    # Add node features
                    for ntype, features in node_features.items():
                        if g.num_nodes(ntype) > 0:
                            g.nodes[ntype].data['feat'] = features
                    
                    # Add edge features
                    for src_type, e_type, dst_type in available_etypes:
                        if e_type in edge_features and g.num_edges(e_type) > 0:
                            g.edges[e_type].data['feat'] = edge_features[e_type]
                    
                    # Add graph-level label
                    if 'task' in g.ntypes and 'next_task' in df.columns:
                        task_labels = torch.tensor(case_group["next_task"].values, dtype=torch.long)
                        if len(task_labels) == g.number_of_nodes('task'):
                            g.nodes['task'].data['label'] = task_labels
                    
                    # Add metadata
                    het_graphs.append(g)
                else:
                    # No edges, create simple graph with just nodes
                    if verbose:
                        logger.warning(f"No valid edges found for case {case_id}, creating node-only graph")
                    
                    # Create a minimal graph with just one node type
                    g = dgl.heterograph({("task", "self", "task"): ([], [])})
                    
                    # Add node features for task type
                    if "task" in node_features:
                        g.nodes['task'].data['feat'] = node_features["task"]
                        if 'next_task' in case_group.columns:
                            g.nodes['task'].data['label'] = torch.tensor(case_group["next_task"].values, dtype=torch.long)
                    
                    het_graphs.append(g)
            else:
                # No valid nodes or edges, log warning
                if verbose:
                    logger.warning(f"No valid nodes or edges found for case {case_id}")
        
        # Update progress
        progress_tracker(batch_end - i)
    
    # Log completion
    if verbose:
        logger.info(f"Built {len(het_graphs)} heterogeneous graphs in {time.time() - start_time:.2f}s")
    
    return het_graphs