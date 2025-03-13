"""
Data loading and processing utilities for DGL-based graph models.
"""

import torch
import dgl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

def collate_dgl_graphs(graphs):
    """
    Collate function for batching DGL graphs
    
    Args:
        graphs: List of DGL graphs to batch
        
    Returns:
        Batched DGL graph
    """
    return dgl.batch(graphs)

def get_graph_dataloader(graphs, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for DGL graphs
    
    Args:
        graphs: List of DGL graphs
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for DGL graphs
    """
    from dgl.dataloading import GraphDataLoader
    
    return GraphDataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dgl_graphs
    )

def get_graph_targets(g):
    """
    Extract graph-level targets from a DGL graph or batched graph
    
    Args:
        g: DGL graph or batched graph
        
    Returns:
        Graph-level target tensor
    """
    if 'label' in g.ndata:
        # Get node labels
        node_labels = g.ndata['label']
        
        if hasattr(g, 'batch_size') and g.batch_size > 1:
            # For batched graph, extract one label per graph
            batch_num_nodes = g.batch_num_nodes()
            graph_labels = []
            
            node_offset = 0
            for num_nodes in batch_num_nodes:
                # Get labels for this graph
                graph_node_labels = node_labels[node_offset:node_offset + num_nodes]
                
                # Use mode (most common label) as graph label
                if len(graph_node_labels) > 0:
                    values, counts = torch.unique(graph_node_labels, return_counts=True)
                    mode_idx = torch.argmax(counts)
                    graph_labels.append(values[mode_idx])
                else:
                    # Fallback if no labels
                    graph_labels.append(torch.tensor(0, device=node_labels.device))
                
                # Update offset
                node_offset += num_nodes
            
            return torch.stack(graph_labels)
        else:
            # For single graph, use mode of node labels
            values, counts = torch.unique(node_labels, return_counts=True)
            mode_idx = torch.argmax(counts)
            return values[mode_idx].unsqueeze(0)
    else:
        # No labels found
        return None

def get_batch_graphs_from_indices(graphs, indices):
    """
    Get a list of graphs from indices
    
    Args:
        graphs: List of DGL graphs
        indices: List of indices to extract
        
    Returns:
        List of DGL graphs at the specified indices
    """
    return [graphs[i] for i in indices]

def apply_to_nodes(g, func):
    """
    Apply a function to all nodes in a graph
    
    Args:
        g: DGL graph
        func: Function to apply to node features
        
    Returns:
        Updated graph
    """
    # Create a new graph to avoid modifying the original
    new_g = g.clone()
    
    # Apply function to node features
    if 'feat' in g.ndata:
        new_g.ndata['feat'] = func(g.ndata['feat'])
    
    return new_g

def apply_to_edges(g, func):
    """
    Apply a function to all edges in a graph
    
    Args:
        g: DGL graph
        func: Function to apply to edge features
        
    Returns:
        Updated graph
    """
    # Create a new graph to avoid modifying the original
    new_g = g.clone()
    
    # Apply function to edge features
    if 'feat' in g.edata:
        new_g.edata['feat'] = func(g.edata['feat'])
    
    return new_g

def create_node_masks(g, mask_ratio=0.1):
    """
    Create node feature masks for self-supervised learning
    
    Args:
        g: DGL graph
        mask_ratio: Ratio of nodes to mask
        
    Returns:
        Graph with added mask
    """
    num_nodes = g.num_nodes()
    mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_ratio)]
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_indices] = True
    
    # Add mask to graph
    g.ndata['mask'] = mask
    
    # Store original features for masked nodes
    if 'feat' in g.ndata:
        g.ndata['orig_feat'] = g.ndata['feat'].clone()
        
        # Apply masking (replace with zeros)
        masked_feat = g.ndata['feat'].clone()
        masked_feat[mask] = 0.0
        g.ndata['feat'] = masked_feat
    
    return g

def convert_batch_to_graphs(batch_data):
    """
    Convert batch of arbitrary graph data to DGL graphs
    
    Args:
        batch_data: Batch data that needs to be converted to DGL graphs
        
    Returns:
        List of DGL graphs
    """
    graphs = []
    
    # Handle different input formats
    if hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index'):
        # Handle PyG-style data
        # This is a backward compatibility feature during migration
        if hasattr(batch_data, 'batch'):
            # This is a batched PyG graph, decompose it into individual graphs
            batch_size = batch_data.batch.max().item() + 1
            for i in range(batch_size):
                mask = batch_data.batch == i
                node_indices = torch.where(mask)[0]
                
                # Get features for this graph
                x = batch_data.x[mask]
                
                # Get edges for this graph
                # Need to remap edge indices to local indices
                edge_mask = torch.isin(batch_data.edge_index[0], node_indices) & \
                           torch.isin(batch_data.edge_index[1], node_indices)
                
                if edge_mask.sum() > 0:
                    edge_index = batch_data.edge_index[:, edge_mask]
                    
                    # Remap node indices
                    idx_map = {idx.item(): new_idx for new_idx, idx in enumerate(node_indices)}
                    edge_index_remapped = torch.tensor([
                        [idx_map[idx.item()] for idx in edge_index[0]],
                        [idx_map[idx.item()] for idx in edge_index[1]]
                    ])
                    
                    # Create DGL graph
                    g = dgl.graph((edge_index_remapped[0], edge_index_remapped[1]), num_nodes=len(node_indices))
                    
                    # Add node features
                    g.ndata['feat'] = x
                    
                    # Add node labels if available
                    if hasattr(batch_data, 'y') and batch_data.y is not None:
                        g.ndata['label'] = batch_data.y[mask]
                    
                    # Add edge features if available
                    if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
                        edge_attr = batch_data.edge_attr[edge_mask]
                        g.edata['feat'] = edge_attr
                    
                    graphs.append(g)
                else:
                    # Create an empty graph with just nodes
                    g = dgl.graph(([], []), num_nodes=len(node_indices))
                    g.ndata['feat'] = x
                    
                    if hasattr(batch_data, 'y') and batch_data.y is not None:
                        g.ndata['label'] = batch_data.y[mask]
                        
                    graphs.append(g)
        else:
            # Single PyG graph, convert to DGL
            g = dgl.graph((batch_data.edge_index[0], batch_data.edge_index[1]), num_nodes=batch_data.x.shape[0])
            
            # Add node features
            g.ndata['feat'] = batch_data.x
            
            # Add node labels if available
            if hasattr(batch_data, 'y') and batch_data.y is not None:
                g.ndata['label'] = batch_data.y
            
            # Add edge features if available
            if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
                g.edata['feat'] = batch_data.edge_attr
                
            graphs.append(g)
    elif isinstance(batch_data, list):
        # Assuming list of DGL graphs
        return batch_data
    elif isinstance(batch_data, dgl.DGLGraph):
        # Already a DGL graph
        return [batch_data]
    else:
        raise ValueError(f"Unsupported batch data type: {type(batch_data)}")
    
    return graphs

def prepare_graph_data(batch_data, device=None):
    """
    Prepare graph data for model processing
    
    Args:
        batch_data: Input graph data in various formats
        device: Device to move data to
        
    Returns:
        DGL graph or batched graph on specified device
    """
    # Handle device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to DGL graphs if needed
    if not isinstance(batch_data, dgl.DGLGraph):
        graphs = convert_batch_to_graphs(batch_data)
        if len(graphs) > 1:
            batch_data = dgl.batch(graphs)
        else:
            batch_data = graphs[0]
    
    # Move graph to device
    batch_data = batch_data.to(device)
    
    return batch_data