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
    

class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader for DGL graphs with dynamic batch sizing
    Based on the improvement plan but adapted for DGL
    """
    def __init__(
        self, 
        dataset, 
        batch_size=32, 
        shuffle=True, 
        pin_memory=True,
        prefetch_factor=2, 
        memory_threshold=0.85,
        drop_last=False,
        collate_fn=None
    ):
        """
        Initialize memory-efficient data loader
        
        Args:
            dataset: Dataset or list of DGL graphs
            batch_size: Initial batch size
            shuffle: Whether to shuffle the data
            pin_memory: Whether to use pinned memory
            prefetch_factor: Prefetch factor for asynchronous loading
            memory_threshold: Memory usage threshold to trigger adjustments
            drop_last: Whether to drop the last batch if incomplete
            collate_fn: Custom collate function (defaults to dgl.batch)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.memory_threshold = memory_threshold
        self.drop_last = drop_last
        
        # Default to DGL batch collate if not provided
        self.collate_fn = collate_fn or (lambda graphs: dgl.batch(graphs))
        
        # Calculate initial indices
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        
        # Track memory usage for adaptive batch sizing
        self.batch_size_history = []
        self.memory_usage_history = []
        
        # Track current position
        self.position = 0
    
    def __iter__(self):
        """Create iterator for batches"""
        # Reset position
        self.position = 0
        
        # Reshuffle if needed
        if self.shuffle:
            random.shuffle(self.indices)
        
        return self
    
    def __next__(self):
        """Get next batch with memory-aware adaptive sizing"""
        if self.position >= len(self.indices):
            raise StopIteration
        
        # Check current memory usage
        current_memory = self._get_memory_usage()
        
        # Adjust batch size if needed
        current_batch_size = self._adjust_batch_size(current_memory)
        
        # Calculate end position for this batch
        end_position = min(self.position + current_batch_size, len(self.indices))
        
        # Handle drop_last
        if self.drop_last and end_position - self.position < current_batch_size:
            self.position = len(self.indices)  # Move to end
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.position:end_position]
        
        # Update position for next batch
        self.position = end_position
        
        # Track memory and batch size for analysis
        self.batch_size_history.append(current_batch_size)
        self.memory_usage_history.append(current_memory)
        
        # Extract and collate batch
        batch = [self.dataset[i] for i in batch_indices]
        
        # Run garbage collection if memory usage is high
        if current_memory > self.memory_threshold * 0.95:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.collate_fn(batch)
    
    def __len__(self):
        """Get number of batches"""
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def _get_memory_usage(self):
        """Get current memory usage as a fraction"""
        if torch.cuda.is_available():
            # GPU memory
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        else:
            # CPU memory
            vm = psutil.virtual_memory()
            return vm.percent / 100.0
    
    def _adjust_batch_size(self, memory_usage):
        """
        Dynamically adjust batch size based on memory usage
        
        Args:
            memory_usage: Current memory usage as fraction
            
        Returns:
            Adjusted batch size
        """
        if memory_usage > self.memory_threshold:
            # Memory usage too high, reduce batch size
            reduction_factor = self.memory_threshold / max(memory_usage, 0.001)
            new_batch_size = max(1, int(self.batch_size * reduction_factor))
            return new_batch_size
        elif memory_usage < self.memory_threshold * 0.7:
            # Memory usage is low, can increase batch size
            # Be conservative with increases (10% at a time)
            increase_factor = min(1.1, self.memory_threshold / max(memory_usage, 0.001))
            new_batch_size = min(len(self.dataset), int(self.batch_size * increase_factor))
            return new_batch_size
        else:
            # Memory usage is in a good range, keep current batch size
            return self.batch_size
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        return {
            "batch_sizes": self.batch_size_history,
            "memory_usage": self.memory_usage_history,
            "average_batch_size": sum(self.batch_size_history) / max(1, len(self.batch_size_history)),
            "min_batch_size": min(self.batch_size_history) if self.batch_size_history else self.batch_size,
            "max_batch_size": max(self.batch_size_history) if self.batch_size_history else self.batch_size,
            "current_memory": self._get_memory_usage()
        }


def adaptive_normalization(features, feature_statistics=None):
    """
    Apply appropriate normalization based on data characteristics
    Implementation of the normalization reconciliation from the improvement plan
    
    Args:
        features: Feature tensor or array to normalize
        feature_statistics: Optional pre-computed statistics
        
    Returns:
        Normalized features
    """
    # Convert to numpy if tensor
    is_tensor = torch.is_tensor(features)
    if is_tensor:
        device = features.device
        features_np = features.cpu().numpy()
    else:
        features_np = features
    
    # Calculate statistics if not provided
    if feature_statistics is None:
        feature_statistics = {
            'mean': np.mean(features_np, axis=0),
            'std': np.std(features_np, axis=0),
            'min': np.min(features_np, axis=0),
            'max': np.max(features_np, axis=0),
            'skewness': _calculate_skewness(features_np)
        }
    
    # Get statistics
    skewness = feature_statistics['skewness']
    min_vals = feature_statistics['min']
    max_vals = feature_statistics['max']
    
    # Calculate range ratio (avoiding division by zero)
    epsilon = 1e-8
    range_ratio = np.divide(
        max_vals, 
        np.maximum(min_vals, epsilon),
        out=np.ones_like(max_vals),
        where=min_vals>epsilon
    )
    
    # Choose normalization strategy based on data properties
    if np.any(np.abs(skewness) > 1.5) or np.any(range_ratio > 10):
        # Highly skewed with large range differences - use robust scaling
        median = np.median(features_np, axis=0)
        q1 = np.percentile(features_np, 25, axis=0)
        q3 = np.percentile(features_np, 75, axis=0)
        iqr = q3 - q1
        # Avoid division by zero
        iqr = np.maximum(iqr, epsilon)
        normalized = (features_np - median) / iqr
    elif np.any(np.abs(features_np) > 5.0):
        # Large magnitudes - use L2 normalization
        norms = np.sqrt(np.sum(features_np**2, axis=1, keepdims=True))
        norms = np.maximum(norms, epsilon)  # Avoid division by zero
        normalized = features_np / norms
    else:
        # Well-behaved features - use MinMax
        min_vals = feature_statistics['min']
        max_vals = feature_statistics['max']
        range_vals = np.maximum(max_vals - min_vals, epsilon)
        normalized = (features_np - min_vals) / range_vals
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        normalized = torch.tensor(normalized, dtype=features.dtype, device=device)
    
    return normalized

def _calculate_skewness(arr):
    """Calculate skewness of array elements along first axis"""
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    # Calculate skewness (third moment)
    n = arr.shape[0]
    m3 = np.sum((arr - mean)**3, axis=0) / n
    return m3 / (std**3)

def get_pyg_compatible_attributes(g):
    """
    Create PyG-compatible attributes from a DGL graph for backward compatibility
    
    Args:
        g: DGL graph
        
    Returns:
        Dictionary with PyG-style attributes
    """
    compat_dict = {}
    
    # Basic properties
    compat_dict['num_nodes'] = g.num_nodes()
    compat_dict['num_edges'] = g.num_edges()
    
    # Node features
    if 'feat' in g.ndata:
        compat_dict['x'] = g.ndata['feat']
    
    # Edge list (PyG style)
    src, dst = g.edges()
    compat_dict['edge_index'] = torch.stack([src, dst], dim=0)
    
    # Edge features
    if 'feat' in g.edata:
        compat_dict['edge_attr'] = g.edata['feat']
    
    # Labels
    if 'label' in g.ndata:
        compat_dict['y'] = g.ndata['label']
    
    # Batch information if available
    if hasattr(g, 'batch_num_nodes'):
        # Create PyG-style batch tensor
        batch = []
        for i, num_nodes in enumerate(g.batch_num_nodes()):
            batch.extend([i] * num_nodes)
        compat_dict['batch'] = torch.tensor(batch, device=g.device)
    
    return compat_dict