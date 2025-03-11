#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized model implementations with memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import gc
import numpy as np

# Import memory optimizer
from memory_optimizer import MemoryOptimizer


class OptimizedMLP(nn.Module):
    """
    Memory-optimized MLP model for process mining
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Use ModuleList for better memory management
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Print model size estimate
        model_size = MemoryOptimizer.check_model_size(self)
        print(f"MLP model size: {model_size['param_count']:,} parameters ({model_size['param_size_mb']:.2f} MB)")
    
    def forward(self, x):
        # Handle graph data inputs (compatibility with GAT models)
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            # Extract node features
            node_features = x.x
            batch = x.batch
            
            # Apply layers to each node
            for i, layer in enumerate(self.layers):
                node_features = layer(node_features)
                node_features = self.batch_norms[i](node_features)
                node_features = F.relu(node_features)
                node_features = F.dropout(node_features, p=self.dropout, training=self.training)
            
            # Global pooling to get graph-level representations
            pooled = global_mean_pool(node_features, batch)
            
            # Output layer
            return self.output_layer(pooled)
        else:
            # Standard tensor input
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            return self.output_layer(x)


class OptimizedLSTM(nn.Module):
    """
    Memory-optimized LSTM model for next activity prediction
    """
    def __init__(self, num_classes, embedding_dim=64, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Use smaller embeddings for large vocabularies
        self.embedding = nn.Embedding(
            num_embeddings=num_classes + 1,  # +1 for padding
            embedding_dim=embedding_dim,
            padding_idx=0,
            sparse=True  # Use sparse gradients for memory efficiency
        )
        
        # Use optimized LSTM implementation
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Simpler unidirectional LSTM
        )
        
        # Efficient batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Print model size estimate
        model_size = MemoryOptimizer.check_model_size(self)
        print(f"LSTM model size: {model_size['param_count']:,} parameters ({model_size['param_size_mb']:.2f} MB)")
    
    def forward(self, x, seq_len=None):
        # Handle tensor or Data object input for compatibility
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            # Convert graph data to sequence
            # This is a simplification - actual implementation would depend on how
            # sequence data is represented in the graph
            batch_size = torch.unique(x.batch).size(0)
            return self._forward_tensor(x.x.view(batch_size, -1, x.x.size(1)))
        
        # Regular tensor input
        return self._forward_tensor(x, seq_len)
    
    def _forward_tensor(self, x, seq_len=None):
        # Apply embedding for integer inputs (task IDs)
        if x.dtype == torch.long:
            x = self.embedding(x)
        
        # Process with LSTM
        if seq_len is not None:
            # Pack sequence for more efficient processing
            seq_len_sorted, perm_idx = seq_len.sort(0, descending=True)
            x_sorted = x[perm_idx]
            
            # Apply embedding
            if x_sorted.dtype == torch.long:
                x_sorted = self.embedding(x_sorted)
            
            # Pack sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # Process with LSTM
            _, (h_n, _) = self.lstm(packed)
            
            # Get last hidden state
            last_hidden = h_n[-1]
            
            # Reorder to original order
            _, unperm_idx = perm_idx.sort(0)
            last_hidden = last_hidden[unperm_idx]
        else:
            # Process without packing
            output, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]
        
        # Apply batch normalization and dropout
        last_hidden = self.batch_norm(last_hidden)
        last_hidden = self.dropout_layer(last_hidden)
        
        # Final prediction
        return self.fc(last_hidden)


class OptimizedGAT(nn.Module):
    """
    Memory-optimized Graph Attention Network for process mining
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Use ModuleList for better memory management
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(
            in_channels=input_dim, 
            out_channels=hidden_dim,
            heads=heads, 
            dropout=dropout
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim,
                heads=heads, 
                dropout=dropout
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        
        # Initialize with more efficient weight initialization
        self._init_weights()
        
        # Print model size estimate
        model_size = MemoryOptimizer.check_model_size(self)
        print(f"GAT model size: {model_size['param_count']:,} parameters ({model_size['param_size_mb']:.2f} MB)")
    
    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index=None, batch=None):
        # Handle both direct tensor inputs and Data objects
        if edge_index is None and hasattr(x, 'edge_index'):
            edge_index = x.edge_index
        
        if batch is None and hasattr(x, 'batch'):
            batch = x.batch
        
        if hasattr(x, 'x'):
            x = x.x
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        return self.fc(x)


class OptimizedPositionalGAT(nn.Module):
    """
    Memory-optimized Positional Graph Attention Network
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pos_dim=16, 
                 num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pos_dim = pos_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # Positional encoding
        self.pos_encoder = nn.Linear(2, pos_dim)
        
        # Enhanced input dimension with positional encoding
        enhanced_input_dim = input_dim + pos_dim
        
        # Use ModuleList for better memory management
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(
            in_channels=enhanced_input_dim, 
            out_channels=hidden_dim,
            heads=heads, 
            dropout=dropout
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim,
                heads=heads, 
                dropout=dropout
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        
        # Print model size estimate
        model_size = MemoryOptimizer.check_model_size(self)
        print(f"Positional GAT model size: {model_size['param_count']:,} parameters ({model_size['param_size_mb']:.2f} MB)")
    
    def forward(self, data):
        # Extract components from data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Generate positional features
        pos = self._generate_positions(batch, x.size(0))
        pos_embedding = self.pos_encoder(pos)
        
        # Concatenate with input features
        x = torch.cat([x, pos_embedding], dim=1)
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        return self.fc(x)
    
    def _generate_positions(self, batch, num_nodes):
        """Generate positional features for nodes"""
        device = batch.device
        pos = torch.zeros((num_nodes, 2), device=device)
        
        # Get unique batches and counts
        unique_batches, counts = torch.unique(batch, return_counts=True)
        
        # Generate grid-like positions for each graph
        start_idx = 0
        for b_idx, count in zip(unique_batches, counts):
            # Create a grid
            grid_size = int(np.ceil(np.sqrt(count.item())))
            for i in range(count.item()):
                row = i // grid_size
                col = i % grid_size
                pos[start_idx + i, 0] = row / max(grid_size - 1, 1)
                pos[start_idx + i, 1] = col / max(grid_size - 1, 1)
            start_idx += count.item()
        
        return pos


class OptimizedDiverseGAT(nn.Module):
    """
    Memory-optimized Diverse Graph Attention Network
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 heads=4, dropout=0.5, diversity_weight=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.diversity_weight = diversity_weight
        
        # Use ModuleList for better memory management
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Head attention projections (for diversity loss)
        self.attention_projectors = nn.ParameterList([
            nn.Parameter(torch.Tensor(heads, hidden_dim))
            for _ in range(num_layers)
        ])
        
        # Input layer
        self.convs.append(GATConv(
            in_channels=input_dim, 
            out_channels=hidden_dim,
            heads=heads, 
            dropout=dropout
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim,
                heads=heads, 
                dropout=dropout
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * heads, output_dim)
        
        # Initialize projectors with orthogonal initialization
        self._init_projectors()
        
        # Print model size estimate
        model_size = MemoryOptimizer.check_model_size(self)
        print(f"Diverse GAT model size: {model_size['param_count']:,} parameters ({model_size['param_size_mb']:.2f} MB)")
    
    def _init_projectors(self):
        """Initialize attention projectors with orthogonal matrices"""
        for projector in self.attention_projectors:
            nn.init.orthogonal_(projector)
    
    def forward(self, x, edge_index=None, batch=None):
        # Handle both direct tensor inputs and Data objects
        if edge_index is None and hasattr(x, 'edge_index'):
            edge_index = x.edge_index
        
        if batch is None and hasattr(x, 'batch'):
            batch = x.batch
        
        if hasattr(x, 'x'):
            x = x.x
        
        # Track diversity loss
        diversity_loss = 0.0
        
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            # Standard convolution
            x = conv(x, edge_index)
            
            # Calculate diversity loss for this layer
            layer_div_loss = self._calculate_diversity_loss(x, i)
            diversity_loss += layer_div_loss
            
            # Apply normalization, activation, and dropout
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Output layer
        logits = self.fc(x)
        
        # Scale diversity loss
        diversity_loss = self.diversity_weight * diversity_loss
        
        return logits, diversity_loss
    
    def _calculate_diversity_loss(self, x, layer_idx):
        """Calculate diversity loss between attention heads"""
        if self.heads <= 1:
            return torch.tensor(0.0, device=x.device)
        
        # Reshape x into [num_nodes, heads, hidden_dim]
        x_reshaped = x.view(-1, self.heads, self.hidden_dim)
        
        # Get projectors for this layer
        projectors = self.attention_projectors[layer_idx]
        
        # Project each head's output
        projected = torch.bmm(x_reshaped, projectors.unsqueeze(0).expand(x_reshaped.size(0), -1, -1))
        
        # Calculate pairwise cosine similarity between head outputs
        sim_loss = 0.0
        num_comparisons = 0
        
        for i in range(self.heads):
            for j in range(i+1, self.heads):
                head_i = projected[:, i, :]
                head_j = projected[:, j, :]
                
                # Calculate cosine similarity
                cos_sim = F.cosine_similarity(head_i, head_j, dim=1).mean()
                
                # Square to penalize both positive and negative correlations
                sim_loss += cos_sim ** 2
                num_comparisons += 1
        
        # Normalize by number of comparisons
        if num_comparisons > 0:
            sim_loss = sim_loss / num_comparisons
        
        return sim_loss


class OptimizedEnhancedGNN(nn.Module):
    """
    Memory-optimized Enhanced GNN combining positional encoding and diverse attention
    """
    def __init__(self, input_dim, hidden_dim, output_dim, pos_dim=16, num_layers=2,
                 heads=4, dropout=0.5, diversity_weight=0.1, predict_time=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pos_dim = pos_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.diversity_weight = diversity_weight
        self.predict_time = predict_time
        
        # Positional encoding
        self.pos_encoder = nn.Linear(2, pos_dim)
        
        # Enhanced input dimension with positional encoding
        enhanced_input_dim = input_dim + pos_dim
        
        # Use ModuleList for better memory management
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        
        # Head attention projectors (for diversity loss)
        self.attention_projectors = nn.ParameterList([
            nn.Parameter(torch.Tensor(heads, hidden_dim))
            for _ in range(num_layers)
        ])
        
        # First layer
        self.convs.append(GATConv(
            in_channels=enhanced_input_dim, 
            out_channels=hidden_dim,
            heads=heads, 
            dropout=dropout
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            # Residual projection if shapes don't match
            self.residual_projs.append(nn.Linear(hidden_dim * heads, hidden_dim * heads))
            
            self.convs.append(GATConv(
                in_channels=hidden_dim * heads, 
                out_channels=hidden_dim,
                heads=heads, 
                dropout=dropout
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layers
        self.task_pred = nn.Linear(hidden_dim * heads, output_dim)
        
        # Time prediction head (optional)
        if predict_time:
            self.time_pred = nn.Sequential(
                nn.Linear(hidden_dim * heads, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Initialize projectors
        self._init_projectors()
        
        # Print model size estimate
        model_size = MemoryOptimizer.check_model_size(self)
        print(f"Enhanced GNN model size: {model_size['param_count']:,} parameters ({model_size['param_size_mb']:.2f} MB)")
    
    def _init_projectors(self):
        """Initialize attention projectors with orthogonal matrices"""
        for projector in self.attention_projectors:
            nn.init.orthogonal_(projector)
    
    def forward(self, data):
        # Extract components from data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Generate positional features
        pos = self._generate_positions(batch, x.size(0))
        pos_embedding = self.pos_encoder(pos)
        
        # Concatenate with input features
        x = torch.cat([x, pos_embedding], dim=1)
        
        # Track diversity loss
        diversity_loss = 0.0
        
        # Apply GAT layers
        prev_x = None
        for i, conv in enumerate(self.convs):
            # Save input for residual connection
            if i > 0:
                prev_x = x
            
            # Standard convolution
            x = conv(x, edge_index)
            
            # Calculate diversity loss for this layer
            layer_div_loss = self._calculate_diversity_loss(x, i)
            diversity_loss += layer_div_loss
            
            # Apply normalization and activation
            x = self.batch_norms[i](x)
            x = F.elu(x)
            
            # Apply residual connection for hidden layers
            if i > 0 and prev_x is not None:
                x = x + self.residual_projs[i-1](prev_x)
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Task prediction
        task_pred = self.task_pred(x)
        
        # Time prediction (if enabled)
        time_pred = self.time_pred(x) if self.predict_time else None
        
        # Scale diversity loss
        diversity_loss = self.diversity_weight * diversity_loss
        
        # Return as dictionary
        return {
            'task_pred': task_pred,
            'time_pred': time_pred,
            'diversity_loss': diversity_loss,
            'graph_embeddings': x
        }
    
    def _generate_positions(self, batch, num_nodes):
        """Generate positional features for nodes"""
        device = batch.device
        pos = torch.zeros((num_nodes, 2), device=device)
        
        # Get unique batches and counts
        unique_batches, counts = torch.unique(batch, return_counts=True)
        
        # Generate grid-like positions for each graph
        start_idx = 0
        for b_idx, count in zip(unique_batches, counts):
            # Create a grid
            grid_size = int(np.ceil(np.sqrt(count.item())))
            for i in range(count.item()):
                row = i // grid_size
                col = i % grid_size
                pos[start_idx + i, 0] = row / max(grid_size - 1, 1)
                pos[start_idx + i, 1] = col / max(grid_size - 1, 1)
            start_idx += count.item()
        
        return pos
    
    def _calculate_diversity_loss(self, x, layer_idx):
        """Calculate diversity loss between attention heads"""
        if self.heads <= 1:
            return torch.tensor(0.0, device=x.device)
        
        # Reshape x into [num_nodes, heads, hidden_dim]
        x_reshaped = x.view(-1, self.heads, self.hidden_dim)
        
        # Get projectors for this layer
        projectors = self.attention_projectors[layer_idx]
        
        # Project each head's output
        projected = torch.bmm(x_reshaped, projectors.unsqueeze(0).expand(x_reshaped.size(0), -1, -1))
        
        # Calculate pairwise cosine similarity between head outputs
        sim_loss = 0.0
        num_comparisons = 0
        
        for i in range(self.heads):
            for j in range(i+1, self.heads):
                head_i = projected[:, i, :]
                head_j = projected[:, j, :]
                
                # Calculate cosine similarity
                cos_sim = F.cosine_similarity(head_i, head_j, dim=1).mean()
                
                # Square to penalize both positive and negative correlations
                sim_loss += cos_sim ** 2
                num_comparisons += 1
        
        # Normalize by number of comparisons
        if num_comparisons > 0:
            sim_loss = sim_loss / num_comparisons
        
        return sim_loss


# Factory function to create optimized models
def create_optimized_model(model_type, input_dim, hidden_dim, output_dim, **kwargs):
    """
    Create an optimized model based on type
    
    Args:
        model_type: Type of model to create
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        **kwargs: Additional model-specific parameters
    
    Returns:
        Instantiated model
    """
    # Collect garbage before creating model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if model_type == 'mlp':
        hidden_dims = kwargs.get('hidden_dims', [hidden_dim, hidden_dim // 2])
        dropout = kwargs.get('dropout', 0.3)
        return OptimizedMLP(input_dim, hidden_dims, output_dim, dropout)
    
    elif model_type == 'lstm':
        num_classes = output_dim
        embedding_dim = kwargs.get('embedding_dim', hidden_dim)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.3)
        return OptimizedLSTM(num_classes, embedding_dim, hidden_dim, num_layers, dropout)
    
    elif model_type == 'basic_gat':
        num_layers = kwargs.get('num_layers', 2)
        heads = kwargs.get('heads', 4)
        dropout = kwargs.get('dropout', 0.5)
        return OptimizedGAT(input_dim, hidden_dim, output_dim, num_layers, heads, dropout)
    
    elif model_type == 'positional_gat':
        pos_dim = kwargs.get('pos_dim', 16)
        num_layers = kwargs.get('num_layers', 2)
        heads = kwargs.get('heads', 4)
        dropout = kwargs.get('dropout', 0.5)
        return OptimizedPositionalGAT(input_dim, hidden_dim, output_dim, pos_dim, num_layers, heads, dropout)
    
    elif model_type == 'diverse_gat':
        num_layers = kwargs.get('num_layers', 2)
        heads = kwargs.get('heads', 4)
        dropout = kwargs.get('dropout', 0.5)
        diversity_weight = kwargs.get('diversity_weight', 0.1)
        return OptimizedDiverseGAT(input_dim, hidden_dim, output_dim, num_layers, heads, dropout, diversity_weight)
    
    elif model_type == 'enhanced_gnn':
        pos_dim = kwargs.get('pos_dim', 16)
        num_layers = kwargs.get('num_layers', 2)
        heads = kwargs.get('heads', 4)
        dropout = kwargs.get('dropout', 0.5)
        diversity_weight = kwargs.get('diversity_weight', 0.1)
        predict_time = kwargs.get('predict_time', False)
        return OptimizedEnhancedGNN(input_dim, hidden_dim, output_dim, pos_dim, num_layers, heads, dropout, diversity_weight, predict_time)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")