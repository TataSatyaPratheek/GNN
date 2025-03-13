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
    Get a list of graphs from indices with proper error handling
    
    Args:
        graphs: List of DGL graphs
        indices: List of indices to extract
        
    Returns:
        List of DGL graphs at the specified indices
    """
    if not isinstance(graphs, list):
        raise TypeError(f"Expected list of graphs, got {type(graphs)}")
    
    if len(graphs) == 0:
        raise ValueError("Empty graph list provided")
    
    # Check if indices are valid
    max_idx = max(indices) if indices else -1
    if max_idx >= len(graphs):
        raise IndexError(f"Index {max_idx} out of range for graph list of length {len(graphs)}")
    
    # Extract graphs
    result = [graphs[i] for i in indices]
    
    # Verify all are DGL graphs
    for i, g in enumerate(result):
        if not isinstance(g, dgl.DGLGraph):
            raise TypeError(f"Graph at index {indices[i]} is not a DGL graph: {type(g)}")
    
    return result

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

def apply_to_nodes(g, func):
    """
    Apply a function to all nodes in a graph with DGL optimized approach
    
    Args:
        g: DGL graph
        func: Function to apply to node features
        
    Returns:
        Updated graph
    """
    # Using DGL's in-place feature modification when possible
    if 'feat' in g.ndata:
        # Create a new graph only if necessary
        if func.__code__.co_argcount > 1 or g.is_readonly():
            # Create a new graph to avoid modifying the original
            new_g = g.clone()
            new_g.ndata['feat'] = func(g.ndata['feat'])
            return new_g
        else:
            # Apply function in-place to save memory
            g.ndata['feat'] = func(g.ndata['feat'])
            return g
    return g

def apply_to_edges(g, func):
    """
    Apply a function to all edges in a graph using DGL optimized approach
    
    Args:
        g: DGL graph
        func: Function to apply to edge features
        
    Returns:
        Updated graph
    """
    if 'feat' in g.edata:
        # Use in-place operations when possible
        if func.__code__.co_argcount > 1 or g.is_readonly():
            # Create a new graph only when necessary
            new_g = g.clone()
            new_g.edata['feat'] = func(g.edata['feat'])
            return new_g
        else:
            # Apply function in-place
            g.edata['feat'] = func(g.edata['feat'])
            return g
    return g

def create_node_masks(g, mask_ratio=0.1):
    """
    Create node feature masks for self-supervised learning with DGL-optimized approach
    
    Args:
        g: DGL graph
        mask_ratio: Ratio of nodes to mask
        
    Returns:
        Graph with added mask
    """
    num_nodes = g.num_nodes()
    mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_ratio)]
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=g.device)
    mask[mask_indices] = True
    
    # Add mask to graph
    g.ndata['mask'] = mask
    
    # Store original features for masked nodes if they exist
    if 'feat' in g.ndata:
        # Save original features only for masked nodes to save memory
        orig_feat = g.ndata['feat'].clone()
        g.ndata['orig_feat'] = orig_feat
        
        # Apply masking (replace with zeros) using DGL's efficient indexing
        if mask.any():
            masked_feat = g.ndata['feat'].clone()
            masked_feat[mask] = 0.0
            g.ndata['feat'] = masked_feat
    
    return g

def prepare_graph_batch(batch_data, device=None, convert_pyg=True):
    """
    Prepare a batch of graphs for model processing with enhanced DGL integration
    
    Args:
        batch_data: Graph data in various formats
        device: Target device for the graphs
        convert_pyg: Whether to convert PyG graphs to DGL if detected
        
    Returns:
        DGL graph or batched graph on specified device
    """
    import torch
    import dgl
    
    # Handle device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle DGL graph or batched graph directly
    if isinstance(batch_data, dgl.DGLGraph):
        return batch_data.to(device)
    
    # Handle list of DGL graphs
    if isinstance(batch_data, list) and all(isinstance(g, dgl.DGLGraph) for g in batch_data):
        return dgl.batch(batch_data).to(device)
    
    # Handle PyG data conversion if needed
    if convert_pyg and hasattr(batch_data, 'x') and hasattr(batch_data, 'edge_index'):
        return convert_pyg_to_dgl(batch_data, device)
    
    # Handle other formats by falling back to the conversion function
    graphs = convert_batch_to_graphs(batch_data)
    if len(graphs) > 1:
        return dgl.batch(graphs).to(device)
    elif len(graphs) == 1:
        return graphs[0].to(device)
    else:
        raise ValueError("No valid graphs found in batch data")

def convert_pyg_to_dgl(pyg_data):
    """
    Convert PyTorch Geometric data format to DGL format
    
    Args:
        pyg_data: PyG data object or batch
        
    Returns:
        Equivalent DGL graph or batched graph
    """
    import dgl
    import torch
    
    # Check if input is already a DGL graph
    if isinstance(pyg_data, dgl.DGLGraph):
        return pyg_data
    
    # Check if input has PyG-style attributes
    if hasattr(pyg_data, 'x') and hasattr(pyg_data, 'edge_index'):
        if hasattr(pyg_data, 'batch'):
            # This is a batched PyG graph
            batch_size = pyg_data.batch.max().item() + 1
            dgl_graphs = []
            
            for i in range(batch_size):
                # Extract nodes for this graph
                mask = pyg_data.batch == i
                node_indices = torch.where(mask)[0]
                
                # Get features for this graph
                x = pyg_data.x[mask]
                
                # Get edges for this graph
                edge_mask = torch.isin(pyg_data.edge_index[0], node_indices) & \
                           torch.isin(pyg_data.edge_index[1], node_indices)
                
                if edge_mask.sum() > 0:
                    edge_index = pyg_data.edge_index[:, edge_mask]
                    
                    # Remap node indices to local indices
                    idx_map = {idx.item(): new_idx for new_idx, idx in enumerate(node_indices)}
                    edge_src = torch.tensor([idx_map[idx.item()] for idx in edge_index[0]])
                    edge_dst = torch.tensor([idx_map[idx.item()] for idx in edge_index[1]])
                    
                    # Create DGL graph
                    g = dgl.graph((edge_src, edge_dst), num_nodes=len(node_indices))
                    
                    # Add node features
                    g.ndata['feat'] = x
                    
                    # Add node labels if available
                    if hasattr(pyg_data, 'y') and pyg_data.y is not None:
                        g.ndata['label'] = pyg_data.y[mask]
                    
                    # Add edge features if available
                    if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
                        g.edata['feat'] = pyg_data.edge_attr[edge_mask]
                else:
                    # Create an empty graph with just nodes
                    g = dgl.graph(([], []), num_nodes=len(node_indices))
                    g.ndata['feat'] = x
                    
                    # Add node labels if available
                    if hasattr(pyg_data, 'y') and pyg_data.y is not None:
                        g.ndata['label'] = pyg_data.y[mask]
                
                dgl_graphs.append(g)
            
            # Batch the DGL graphs
            return dgl.batch(dgl_graphs)
        else:
            # Single PyG graph
            g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]), 
                           num_nodes=pyg_data.x.shape[0])
            g.ndata['feat'] = pyg_data.x
            
            # Add node labels if available
            if hasattr(pyg_data, 'y') and pyg_data.y is not None:
                g.ndata['label'] = pyg_data.y
            
            # Add edge features if available
            if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
                g.edata['feat'] = pyg_data.edge_attr
            
            return g
    else:
        raise ValueError("Input data doesn't have PyG format attributes (x, edge_index)")
    """
    Convert a PyG graph to DGL format optimized for memory
    
    Args:
        pyg_data: PyG graph data
        device: Target device
        
    Returns:
        DGL graph or batched graph
    """
    import torch
    import dgl
    
    # Check if this is a batched PyG graph
    if hasattr(pyg_data, 'batch'):
        # Extract batching information
        batch_size = pyg_data.batch.max().item() + 1
        graphs = []
        
        # Process each graph in the batch
        for i in range(batch_size):
            # Get nodes for this graph
            mask = pyg_data.batch == i
            node_indices = torch.where(mask)[0]
            
            # Skip empty graphs
            if len(node_indices) == 0:
                continue
                
            # Get node features
            x = pyg_data.x[mask]
            
            # Get edges for this graph
            edge_mask = torch.isin(pyg_data.edge_index[0], node_indices) & \
                       torch.isin(pyg_data.edge_index[1], node_indices)
            
            # Create DGL graph
            if edge_mask.sum() > 0:
                # Get edges with mapping to local indices
                edge_index = pyg_data.edge_index[:, edge_mask]
                
                # Create node index mapping
                idx_map = {idx.item(): new_idx for new_idx, idx in enumerate(node_indices)}
                
                # Map edge indices efficiently
                src = torch.tensor([idx_map[idx.item()] for idx in edge_index[0]])
                dst = torch.tensor([idx_map[idx.item()] for idx in edge_index[1]])
                
                # Create DGL graph
                g = dgl.graph((src, dst), num_nodes=len(node_indices))
                
                # Add node features
                g.ndata['feat'] = x
                
                # Add node labels if available
                if hasattr(pyg_data, 'y') and pyg_data.y is not None:
                    g.ndata['label'] = pyg_data.y[mask]
                
                # Add edge features if available
                if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
                    g.edata['feat'] = pyg_data.edge_attr[edge_mask]
            else:
                # Create graph with only nodes
                g = dgl.graph(([], []), num_nodes=len(node_indices))
                g.ndata['feat'] = x
                
                # Add node labels if available
                if hasattr(pyg_data, 'y') and pyg_data.y is not None:
                    g.ndata['label'] = pyg_data.y[mask]
            
            graphs.append(g)
        
        # Return batched graph
        return dgl.batch(graphs).to(device) if graphs else None
    else:
        # Single PyG graph
        x = pyg_data.x
        edge_index = pyg_data.edge_index
        
        # Create DGL graph
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
        
        # Add node features
        g.ndata['feat'] = x
        
        # Add node labels if available
        if hasattr(pyg_data, 'y') and pyg_data.y is not None:
            g.ndata['label'] = pyg_data.y
        
        # Add edge features if available
        if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
            g.edata['feat'] = pyg_data.edge_attr
        
        return g.to(device)