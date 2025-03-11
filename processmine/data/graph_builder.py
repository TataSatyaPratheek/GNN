#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph building utilities for process mining
"""

import torch
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import time
import gc
import logging

logger = logging.getLogger(__name__)

def build_graph_data(df):
    """
    Convert preprocessed data into graph format for GNN
    
    Args:
        df: Process data dataframe
        
    Returns:
        List of graph data objects
    """
    print("\n==== Building Graph Data ====")
    start_time = time.time()
    
    graphs = []
    case_groups = df.groupby("case_id")
    num_cases = len(case_groups)
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure feature extraction was performed correctly.")
    
    # Create progress bar
    progress_bar = tqdm(
        case_groups, 
        desc="Building graphs",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100,
        total=num_cases
    )
    
    # Track statistics
    edge_counts = []
    node_counts = []
    
    for cid, cdata in progress_bar:
        cdata = cdata.sort_values("timestamp")

        # Create node features
        x_data = torch.tensor(cdata[feature_cols].values, dtype=torch.float)

        # Create edges between sequential activities
        n_nodes = len(cdata)
        node_counts.append(n_nodes)
        
        if n_nodes > 1:
            src = list(range(n_nodes-1))
            tgt = list(range(1,n_nodes))
            edge_index = torch.tensor([src+tgt, tgt+src], dtype=torch.long)
            edge_counts.append(2 * (n_nodes - 1))  # Bidirectional edges
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_counts.append(0)
            
        y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        data_obj = Data(x=x_data, edge_index=edge_index, y=y_data)
        graphs.append(data_obj)
        
        # Periodically collect garbage to prevent memory buildup
        if len(graphs) % 1000 == 0:
            gc.collect()

    # Report statistics
    avg_nodes = np.mean(node_counts)
    avg_edges = np.mean(edge_counts)
    max_nodes = np.max(node_counts)
    
    print(f"\033[1mGraph Statistics\033[0m:")
    print(f"  Total graphs: \033[96m{len(graphs):,}\033[0m")
    print(f"  Avg nodes per graph: \033[96m{avg_nodes:.2f}\033[0m")
    print(f"  Avg edges per graph: \033[96m{avg_edges:.2f}\033[0m")
    print(f"  Max nodes in a graph: \033[96m{max_nodes}\033[0m")
    print(f"Graphs built in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return graphs

def build_enhanced_graph_data(df):
    """
    Build enhanced graph data with additional edge features
    
    Args:
        df: Process data dataframe
        
    Returns:
        List of enhanced graph data objects
    """
    print("\n==== Building Enhanced Graph Data ====")
    start_time = time.time()
    
    graphs = []
    case_groups = df.groupby("case_id")
    num_cases = len(case_groups)
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        raise ValueError("No feature columns found. Ensure feature extraction was performed correctly.")
    
    # Create progress bar
    progress_bar = tqdm(
        case_groups, 
        desc="Building enhanced graphs",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100,
        total=num_cases
    )
    
    # Track statistics
    edge_counts = []
    node_counts = []
    
    for cid, cdata in progress_bar:
        cdata = cdata.sort_values("timestamp")

        # Create node features
        x_data = torch.tensor(cdata[feature_cols].values, dtype=torch.float)

        # Create edges between activities
        n_nodes = len(cdata)
        node_counts.append(n_nodes)
        
        if n_nodes > 1:
            # Create standard sequential edges
            src = list(range(n_nodes-1))
            tgt = list(range(1,n_nodes))
            
            # Add edge features: time between events
            timestamps = cdata["timestamp"].values
            time_diffs = []
            for i in range(n_nodes-1):
                time_diff = (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                time_diffs.append(time_diff)
            
            # Normalize time differences
            max_time = max(time_diffs) if time_diffs else 1.0
            norm_time_diffs = [t / max_time for t in time_diffs]
            
            # Add bidirectional edges with time features
            edge_index = torch.tensor([src+tgt, tgt+src], dtype=torch.long)
            
            # Edge features: forward edges have time diff, backward edges have -time diff
            edge_attr = torch.tensor(norm_time_diffs + [-t for t in norm_time_diffs], dtype=torch.float).view(-1, 1)
            
            edge_counts.append(2 * (n_nodes - 1))  # Bidirectional edges
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0,1), dtype=torch.float)
            edge_counts.append(0)
            
        y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        
        # Create enhanced graph data object with edge attributes
        data_obj = Data(x=x_data, edge_index=edge_index, edge_attr=edge_attr, y=y_data)
        graphs.append(data_obj)
        
        # Periodically collect garbage to prevent memory buildup
        if len(graphs) % 1000 == 0:
            gc.collect()

    # Report statistics
    avg_nodes = np.mean(node_counts)
    avg_edges = np.mean(edge_counts)
    max_nodes = np.max(node_counts)
    
    print(f"\033[1mEnhanced Graph Statistics\033[0m:")
    print(f"  Total graphs: \033[96m{len(graphs):,}\033[0m")
    print(f"  Avg nodes per graph: \033[96m{avg_nodes:.2f}\033[0m")
    print(f"  Avg edges per graph: \033[96m{avg_edges:.2f}\033[0m")
    print(f"  Max nodes in a graph: \033[96m{max_nodes}\033[0m")
    print(f"Enhanced graphs built in \033[96m{time.time() - start_time:.2f}s\033[0m")

    return graphs