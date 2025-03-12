"""
Memory-efficient graph neural network architectures for process mining with 
optimized attention mechanisms and minimal memory footprint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import logging
import math
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

logger = logging.getLogger(__name__)

class BaseProcessModel(nn.Module):
    """Base class for all process mining models with unified interface"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(self, x):
        """Make predictions with a consistent interface"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get("task_pred", next(iter(outputs.values())))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            _, predictions = torch.max(logits, dim=1)
            return predictions
    
    def get_embeddings(self, x):
        """Get embeddings from model - to be implemented by subclasses"""
        raise NotImplementedError("Embedding extraction not implemented for this model")
    
    def get_attention_weights(self, x):
        """Get attention weights from model - to be implemented by compatible subclasses"""
        raise NotImplementedError("Attention weights not available for this model")
    
    def get_parameter_count(self):
        """Get parameter count for the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def memory_usage(self):
        """Estimate model memory usage in MB"""
        # Calculate parameter memory
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        
        # Calculate buffer memory (e.g., for BatchNorm)
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        
        # Estimate activation memory (rough approximation)
        # This varies by model and batch size
        activation_bytes = param_bytes * 2  # Rough estimate
        
        # Convert to MB
        param_mb = param_bytes / (1024 * 1024)
        buffer_mb = buffer_bytes / (1024 * 1024)
        activation_mb = activation_bytes / (1024 * 1024)
        total_mb = param_mb + buffer_mb + activation_mb
        
        return {
            'parameters_mb': param_mb,
            'buffers_mb': buffer_mb,
            'activations_mb': activation_mb,
            'total_mb': total_mb
        }

class MemoryEfficientGNN(BaseProcessModel):
    """
    Memory-efficient graph neural network for process mining with optimized attention mechanisms
    
    This model includes multiple enhancements:
    1. Checkpointing for reduced memory usage during backpropagation
    2. Sparse attention implementations for efficiency with large graphs
    3. Flexible attention mechanisms (basic, positional, diverse, or combined)
    4. Layer normalization and residual connections for better training dynamics
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        attention_type: str = "basic",
        pos_enc_dim: int = 16,
        diversity_weight: float = 0.1,
        pooling: str = "mean",
        predict_time: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_residual: bool = True,
        sparse_attention: bool = False,
        use_checkpointing: bool = False,
        mem_efficient: bool = True
    ):
        """
        Initialize memory-efficient GNN
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_layers: Number of GNN layers
            heads: Number of attention heads
            dropout: Dropout probability
            attention_type: Type of attention ("basic", "positional", "diverse", "combined")
            pos_enc_dim: Positional encoding dimension
            diversity_weight: Weight for diversity loss
            pooling: Pooling method ("mean", "sum", "max", "combined", "attention")
            predict_time: Whether to predict time in addition to task
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
            sparse_attention: Whether to use sparse attention for very large graphs
            use_checkpointing: Whether to use gradient checkpointing to save memory
            mem_efficient: Whether to use other memory-efficient implementations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.attention_type = attention_type
        self.pos_enc_dim = pos_enc_dim
        self.diversity_weight = diversity_weight
        self.pooling_type = pooling
        self.predict_time = predict_time
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.sparse_attention = sparse_attention
        self.use_checkpointing = use_checkpointing and torch.cuda.is_available()
        self.mem_efficient = mem_efficient
        
        # Enhanced input transformation
        if attention_type == "positional" or attention_type == "combined":
            # Positional encoding will be added to input
            self.pos_encoder = nn.Linear(2, pos_enc_dim)
            input_with_pos = input_dim + pos_enc_dim
        else:
            input_with_pos = input_dim
        
        # Create GNN layers
        self.convs = nn.ModuleList()
        
        # Create input layer
        self.convs.append(self._create_conv_layer(
            input_with_pos if attention_type in ["positional", "combined"] else input_dim,
            hidden_dim,
            first_layer=True
        ))
        
        # Create hidden layers
        for i in range(1, num_layers):
            self.convs.append(self._create_conv_layer(
                hidden_dim * heads, 
                hidden_dim,
                first_layer=False
            ))
        
        # Normalization layers (batch norm or layer norm)
        if use_batch_norm:
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim * heads) for _ in range(num_layers)
            ])
        elif use_layer_norm:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim * heads) for _ in range(num_layers)
            ])
        else:
            self.norms = None
        
        # Set up pooling layer
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "combined":
            # Combined pooling (concatenate mean, max, sum)
            self.pool = self._combined_pooling
            combined_size = hidden_dim * heads * 3
            self.pool_proj = nn.Linear(combined_size, hidden_dim * heads)
        elif pooling == "attention":
            # Attention-based pooling
            self.pool = self._attention_pooling
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_dim * heads, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Prediction heads
        # Task prediction
        self.task_pred = nn.Linear(hidden_dim * heads, output_dim)
        
        # Time prediction (optional)
        if predict_time:
            self.time_pred = nn.Sequential(
                nn.Linear(hidden_dim * heads, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.time_pred = None
        
        # Initialize weights
        self._init_weights()
        
        # Storage for attention weights
        self.attention_weights = None
    
    def _init_weights(self):
        """Initialize weights optimally for better convergence"""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    # For 1D weights (like in LayerNorm)
                    nn.init.ones_(param)
    
    def _create_conv_layer(self, in_channels, out_channels, first_layer=False):
        """Create the appropriate convolutional layer based on attention type"""
        if self.attention_type == "basic":
            return MemoryEfficientGATConv(
                in_channels, 
                out_channels,
                heads=self.heads,
                dropout=self.dropout,
                add_self_loops=True,
                sparse_attention=self.sparse_attention,
                concat=True
            )
        elif self.attention_type == "positional":
            return PositionalGATConv(
                in_channels, 
                out_channels,
                heads=self.heads,
                pos_dim=self.pos_enc_dim,
                dropout=self.dropout,
                concat=True
            )
        elif self.attention_type == "diverse":
            return DiverseGATConv(
                in_channels, 
                out_channels,
                heads=self.heads,
                diversity_weight=self.diversity_weight,
                dropout=self.dropout,
                concat=True
            )
        elif self.attention_type == "combined":
            return CombinedGATConv(
                in_channels, 
                out_channels,
                heads=self.heads,
                pos_dim=self.pos_enc_dim,
                diversity_weight=self.diversity_weight,
                dropout=self.dropout,
                concat=True
            )
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
    
    def _combined_pooling(self, x, batch):
        """Combined pooling (concatenate mean, max, sum)"""
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        combined = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
        return self.pool_proj(combined)
    
    def _attention_pooling(self, x, batch):
        """Attention-based pooling"""
        # Calculate attention scores
        scores = self.pool_attention(x)
        
        # Apply softmax over nodes in each graph
        scores = scores.view(-1, 1)
        
        # Mask for each graph
        from torch_scatter import scatter_softmax
        scores = scatter_softmax(scores, batch, dim=0)
        
        # Weighted sum of node features
        from torch_scatter import scatter_sum
        weighted_x = x * scores
        return scatter_sum(weighted_x, batch, dim=0)
    
    def _generate_positions(self, batch):
        """
        Generate positional encodings for nodes in each graph
        
        Args:
            batch: Batch assignment tensor [num_nodes]
            
        Returns:
            Position tensor [num_nodes, 2]
        """
        # Get device
        device = batch.device
        
        # Get number of nodes
        num_nodes = batch.size(0)
        
        # Get unique batch indices and their counts
        unique_batches, counts = torch.unique(batch, return_counts=True)
        
        # Create positions tensor
        pos = torch.zeros((num_nodes, 2), device=device)
        
        # Calculate position for each node
        start_idx = 0
        for b_idx, count in zip(unique_batches, counts):
            # Get count as Python int
            count_val = count.item()
            
            # Calculate grid dimensions for 2D layout
            grid_size = int(math.sqrt(count_val)) + 1
            
            # Assign positions in grid pattern
            for i in range(count_val):
                row = i // grid_size
                col = i % grid_size
                
                # Normalize to [0,1]
                norm_row = row / max(1, grid_size - 1)
                norm_col = col / max(1, grid_size - 1)
                
                # Assign position
                pos[start_idx + i, 0] = norm_row
                pos[start_idx + i, 1] = norm_col
            
            # Update start index
            start_idx += count_val
        
        return pos
    
    def forward(self, data):
        """
        Forward pass with memory optimizations
        
        Args:
            data: PyG data object with x, edge_index, and batch attributes
            
        Returns:
            Model output (format depends on configuration)
        """
        # Extract features and structure
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Handle positional encodings
        if self.attention_type in ["positional", "combined"]:
            # Generate positions based on batch indices
            pos = self._generate_positions(batch)
            
            # Add position embeddings to input
            pos_embedding = self.pos_encoder(pos)
            x = torch.cat([x, pos_embedding], dim=-1)
        
        # Track diversity losses and attention weights
        diversity_losses = []
        attn_weights_list = []
        
        # Memory for residual connections
        residual = None
        
        # Process through GNN layers with checkpointing if enabled
        for i, conv in enumerate(self.convs):
            # Apply convolutional layer with optional checkpointing
            if self.use_checkpointing and self.training:
                # Use checkpointing to save memory during backprop
                if self.attention_type in ["diverse", "combined"]:
                    # Handle layers that return diversity loss and attention weights
                    x_new, div_loss, attn_weights = torch.utils.checkpoint.checkpoint(
                        conv, x, edge_index, True  # Return attention flag
                    )
                    x = x_new
                    diversity_losses.append(div_loss)
                    attn_weights_list.append(attn_weights)
                else:
                    # Standard layers
                    if hasattr(conv, 'return_attention_weights'):
                        # For layers that support attention weights
                        x_new, attn_weights = torch.utils.checkpoint.checkpoint(
                            lambda a, b: conv(a, b, return_attention_weights=True),
                            x, edge_index
                        )
                        x = x_new
                        attn_weights_list.append(attn_weights)
                    else:
                        # Layers without attention weights
                        x = torch.utils.checkpoint.checkpoint(conv, x, edge_index)
            else:
                # Standard forward pass without checkpointing
                if self.attention_type in ["diverse", "combined"]:
                    # Layers with diversity loss
                    x, div_loss, attn_weights = conv(x, edge_index, return_attention=True)
                    diversity_losses.append(div_loss)
                    attn_weights_list.append(attn_weights)
                else:
                    # Standard or positional convolution
                    if hasattr(conv, 'return_attention_weights'):
                        # Layers with attention weights
                        x, attn_weights = conv(x, edge_index, return_attention_weights=True)
                        attn_weights_list.append(attn_weights)
                    else:
                        # Plain layers
                        x = conv(x, edge_index)
            
            # Apply normalization if enabled
            if self.norms is not None:
                x = self.norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Apply residual connection if enabled
            if self.use_residual and i > 0 and residual is not None:
                if residual.shape == x.shape:
                    x = x + residual
            
            # Store for next residual connection
            residual = x
            
            # Aggressive memory optimization if needed
            if self.mem_efficient and i < self.num_layers - 1 and torch.cuda.is_available():
                # Free memory from intermediate computations for very large graphs
                if x.shape[0] > 10000:  # Only for very large graphs
                    torch.cuda.empty_cache()
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights_list
        
        # Apply global pooling to get graph-level representations
        pooled = self.pool(x, batch)
        
        # Task prediction
        task_pred = self.task_pred(pooled)
        
        # Prepare output based on configuration
        if self.predict_time and self.time_pred is not None:
            # Dual task prediction (task + time)
            time_pred = self.time_pred(pooled).squeeze(-1)
            outputs = {
                "task_pred": task_pred,
                "time_pred": time_pred
            }
        else:
            # Single task prediction
            outputs = {"task_pred": task_pred}
        
        # Add diversity loss if applicable
        if diversity_losses:
            total_div_loss = sum(diversity_losses)
            outputs["diversity_loss"] = total_div_loss
            outputs["diversity_weight"] = self.diversity_weight
        
        return outputs
    
    def get_embeddings(self, data):
        """
        Extract node embeddings from the model
        
        Args:
            data: PyG data object
            
        Returns:
            Node embeddings
        """
        self.eval()
        with torch.no_grad():
            # Extract features and structure
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Handle positional encodings
            if self.attention_type in ["positional", "combined"]:
                pos = self._generate_positions(batch)
                pos_embedding = self.pos_encoder(pos)
                x = torch.cat([x, pos_embedding], dim=-1)
            
            # Process through GNN layers
            for i, conv in enumerate(self.convs):
                # Apply convolutional layer
                if self.attention_type in ["diverse", "combined"]:
                    x, _, _ = conv(x, edge_index, return_attention=True)
                else:
                    if hasattr(conv, 'return_attention_weights'):
                        x, _ = conv(x, edge_index, return_attention_weights=False)
                    else:
                        x = conv(x, edge_index)
                
                # Apply normalization if enabled
                if self.norms is not None:
                    x = self.norms[i](x)
                
                # Apply activation
                x = F.elu(x)
            
            # Return node embeddings and batch assignments
            return x, batch
    
    def get_attention_weights(self, data):
        """
        Get attention weights for interpretability
        
        Args:
            data: PyG data object
            
        Returns:
            List of attention weights for each layer
        """
        # Run forward pass to populate attention weights
        _ = self.forward(data)
        
        # Return stored attention weights
        return self.attention_weights


class MemoryEfficientGATConv(MessagePassing):
    """
    Memory-efficient Graph Attention Layer with sparse attention option
    for processing very large graphs
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 1, 
        concat: bool = True, 
        negative_slope: float = 0.2, 
        dropout: float = 0.0, 
        add_self_loops: bool = True,
        bias: bool = True,
        sparse_attention: bool = False,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.sparse_attention = sparse_attention
        
        # Linear transformations
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention weights
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Forward pass with optional attention weight return
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features or tuple of (features, attention weights)
        """
        # Add self-loops for better message passing
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Linear transformation to project input features
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Memory-efficient message passing with sparse attention if requested
        if self.sparse_attention and x.size(0) > 10000:
            # For very large graphs, use sparse implementation
            return self._sparse_forward(x, edge_index, return_attention_weights)
        else:
            # For regular graphs, use standard implementation
            return self._dense_forward(x, edge_index, return_attention_weights)
    
    def _dense_forward(self, x, edge_index, return_attention_weights):
        """Standard GATConv implementation using dense attention"""
        # Compute attention scores
        # Extract node features for source and target nodes
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Concatenate source and target features for attention
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        
        # Apply LeakyReLU activation
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention scores using softmax (grouped by source node)
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted aggregation of neighbor features
        out = torch.zeros(x.size(0), self.heads, self.out_channels, device=x.device)
        alpha = alpha.view(-1, self.heads, 1)
        out.index_add_(0, row, x_j * alpha)
        
        # Apply concatenation or mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        # Return with attention weights if requested
        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out
    
    def _sparse_forward(self, x, edge_index, return_attention_weights):
        """Sparse implementation for very large graphs"""
        # This implementation uses less memory by performing operations in chunks
        row, col = edge_index
        
        # Process in chunks to save memory
        chunk_size = 10000
        num_edges = row.size(0)
        
        # Initialize output tensor
        out = torch.zeros(x.size(0), self.heads * self.out_channels if self.concat else self.out_channels, 
                         device=x.device)
        
        # Initialize attention weights for return if needed
        if return_attention_weights:
            all_alpha = torch.zeros(num_edges, self.heads, device=x.device)
        
        # Process edges in chunks
        for i in range(0, num_edges, chunk_size):
            end_idx = min(i + chunk_size, num_edges)
            chunk_row, chunk_col = row[i:end_idx], col[i:end_idx]
            
            # Get features for this chunk
            x_i, x_j = x[chunk_row], x[chunk_col]
            
            # Compute attention scores for this chunk
            alpha = torch.cat([x_i, x_j], dim=-1)
            alpha = (alpha * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            
            # We need to normalize across all edges of each source node
            # This requires looking outside the chunk, so we use a custom function
            unique_rows, inverse_rows = torch.unique(chunk_row, return_inverse=True)
            
            # Initialize normalization values
            row_sums = torch.zeros(unique_rows.size(0), self.heads, device=x.device)
            
            # Sum attention scores for each source node
            for h in range(self.heads):
                row_sums.index_add_(0, inverse_rows, alpha[:, h].unsqueeze(1))
            
            # Normalize using computed sums
            normalized_alpha = alpha.clone()
            eps = 1e-8  # Prevent division by zero
            
            for h in range(self.heads):
                normalized_alpha[:, h] = alpha[:, h] / (row_sums[inverse_rows, h].squeeze() + eps)
            
            # Apply dropout
            normalized_alpha = F.dropout(normalized_alpha, p=self.dropout, training=self.training)
            
            # Save for return if needed
            if return_attention_weights:
                all_alpha[i:end_idx] = normalized_alpha
            
            # Apply weighted aggregation for this chunk
            for h in range(self.heads):
                for j in range(end_idx - i):
                    src_idx = chunk_row[j].item()
                    tgt_idx = chunk_col[j].item()
                    weight = normalized_alpha[j, h]
                    
                    # Add weighted features
                    if self.concat:
                        out[src_idx, h * self.out_channels:(h + 1) * self.out_channels] += \
                            weight * x_j[j, h]
                    else:
                        out[src_idx] += weight * x_j[j, h]
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        # Return with attention weights if requested
        if return_attention_weights:
            all_alpha = all_alpha.view(-1, self.heads, 1)
            return out, (edge_index, all_alpha)
        else:
            return out


class PositionalGATConv(MessagePassing):
    """
    Position-enhanced Graph Attention Layer optimized for memory efficiency
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 1, 
        pos_dim: int = 16,
        concat: bool = True, 
        negative_slope: float = 0.2, 
        dropout: float = 0.0, 
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.pos_dim = pos_dim
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Main transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention weights for content
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Positional attention component
        # Calculate positional component size
        pos_input_size = in_channels - pos_dim  # Input size without positional part
        
        # Attention weights for spatial positions
        self.pos_att = nn.Parameter(torch.Tensor(1, heads, 2 * pos_dim))
        
        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.pos_att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention_weights=False):
        """Forward pass with optional attention weights return"""
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Split input into content and positional parts
        content_x = x[:, :-self.pos_dim]
        pos_x = x[:, -self.pos_dim:]
        
        # Linear transformation for content
        content = self.lin(content_x).view(-1, self.heads, self.out_channels)
        
        # Extract node indices
        row, col = edge_index
        src_content, dst_content = content[row], content[col]
        src_pos, dst_pos = pos_x[row], pos_x[col]
        
        # Compute content attention component
        alpha_content = torch.cat([src_content, dst_content], dim=-1)
        alpha_content = (alpha_content * self.att).sum(dim=-1)
        
        # Compute positional attention component
        alpha_pos = torch.cat([
            src_pos.unsqueeze(1).repeat(1, self.heads, 1),
            dst_pos.unsqueeze(1).repeat(1, self.heads, 1)
        ], dim=-1)
        alpha_pos = (alpha_pos * self.pos_att).sum(dim=-1)
        
        # Combine attentions
        alpha = alpha_content + alpha_pos
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted aggregation for messages
        out = self.propagate(edge_index, x=content, alpha=alpha, size=None)
        
        # Apply concat/mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Apply bias
        if self.bias is not None:
            out = out + self.bias
        
        # Return with attention weights if requested
        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out
    
    def message(self, x_j, alpha):
        """Message function for aggregation"""
        # x_j: [E, heads, out_channels], alpha: [E, heads]
        alpha = alpha.unsqueeze(-1)
        return x_j * alpha
    
    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages using sum"""
        return scatter_add(inputs, index, dim=0, dim_size=dim_size)


class DiverseGATConv(MessagePassing):
    """
    GAT layer with attention diversity loss to avoid attention collapse
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 4, 
        concat: bool = True, 
        diversity_weight: float = 0.1, 
        negative_slope: float = 0.2, 
        dropout: float = 0.0, 
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.diversity_weight = diversity_weight
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Main transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention weights
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention=False):
        """Forward pass with diversity loss"""
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Extract node indices
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Compute attention scores
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # Calculate diversity loss (correlation between heads)
        diversity_loss = self._calculate_diversity_loss(alpha)
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted aggregation
        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)
        
        # Apply concat/mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Apply bias
        if self.bias is not None:
            out = out + self.bias
        
        # Return with diversity loss if requested
        if return_attention:
            return out, diversity_loss, alpha
        else:
            return out, diversity_loss
    
    def _calculate_diversity_loss(self, alpha):
        """Calculate diversity loss to encourage different attention patterns"""
        # Initialize correlation matrix
        device = alpha.device
        head_corr = torch.zeros(self.heads, self.heads, device=device)
        
        # Calculate correlation between attention heads
        for h1 in range(self.heads):
            for h2 in range(h1+1, self.heads):
                # Calculate cosine similarity
                a1 = alpha[:, h1]
                a2 = alpha[:, h2]
                
                # Normalize vectors for numerical stability
                a1_norm = F.normalize(a1, p=2, dim=0)
                a2_norm = F.normalize(a2, p=2, dim=0)
                
                # Compute cosine similarity
                corr = torch.sum(a1_norm * a2_norm)
                
                # Store in matrix (symmetrically)
                head_corr[h1, h2] = corr
                head_corr[h2, h1] = corr
        
        # Average correlation - higher means more similar patterns
        diversity_loss = torch.mean(head_corr) * self.diversity_weight
        
        return diversity_loss
    
    def message(self, x_j, alpha):
        """Message function for aggregation"""
        # x_j: [E, heads, out_channels], alpha: [E, heads]
        alpha = alpha.unsqueeze(-1)
        return x_j * alpha
    
    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages using sum"""
        return scatter_add(inputs, index, dim=0, dim_size=dim_size)


class CombinedGATConv(MessagePassing):
    """
    Combined GAT layer with positional encoding and attention diversity
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 4, 
        concat: bool = True, 
        pos_dim: int = 16,
        diversity_weight: float = 0.1, 
        negative_slope: float = 0.2, 
        dropout: float = 0.0, 
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.pos_dim = pos_dim
        self.diversity_weight = diversity_weight
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Main transformation
        content_dim = in_channels - pos_dim  # Dimension without positional part
        self.lin = nn.Linear(content_dim, heads * out_channels, bias=False)
        
        # Attention weights for content
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Positional attention component
        self.pos_att = nn.Parameter(torch.Tensor(1, heads, 2 * pos_dim))
        
        # Bias
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.pos_att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention=False):
        """Forward pass with diversity loss and position-aware attention"""
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Split features into content and position parts
        content_features = x[:, :-self.pos_dim]
        pos_features = x[:, -self.pos_dim:]
        
        # Apply content transformation
        content = self.lin(content_features).view(-1, self.heads, self.out_channels)
        
        # Extract node indices
        row, col = edge_index
        content_i, content_j = content[row], content[col]
        pos_i, pos_j = pos_features[row], pos_features[col]
        
        # Compute content attention component
        alpha_content = torch.cat([content_i, content_j], dim=-1)
        alpha_content = (alpha_content * self.att).sum(dim=-1)
        
        # Compute positional attention component
        alpha_pos = torch.cat([
            pos_i.unsqueeze(1).repeat(1, self.heads, 1),
            pos_j.unsqueeze(1).repeat(1, self.heads, 1)
        ], dim=-1)
        alpha_pos = (alpha_pos * self.pos_att).sum(dim=-1)
        
        # Combine attentions
        alpha = alpha_content + alpha_pos
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # Calculate diversity loss
        diversity_loss = self._calculate_diversity_loss(alpha)
        
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted aggregation
        out = self.propagate(edge_index, x=content, alpha=alpha, size=None)
        
        # Apply concat/mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Apply bias
        if self.bias is not None:
            out = out + self.bias
        
        # Return with diversity loss if requested
        if return_attention:
            return out, diversity_loss, alpha
        else:
            return out, diversity_loss
    
    def _calculate_diversity_loss(self, alpha):
        """Calculate diversity loss to encourage different attention patterns"""
        # Initialize correlation matrix
        device = alpha.device
        head_corr = torch.zeros(self.heads, self.heads, device=device)
        
        # Calculate correlation between attention heads
        for h1 in range(self.heads):
            for h2 in range(h1+1, self.heads):
                # Calculate cosine similarity
                a1 = alpha[:, h1]
                a2 = alpha[:, h2]
                
                # Normalize vectors for numerical stability
                a1_norm = F.normalize(a1, p=2, dim=0)
                a2_norm = F.normalize(a2, p=2, dim=0)
                
                # Compute cosine similarity
                corr = torch.sum(a1_norm * a2_norm)
                
                # Store in matrix (symmetrically)
                head_corr[h1, h2] = corr
                head_corr[h2, h1] = corr
        
        # Average correlation - higher means more similar patterns
        diversity_loss = torch.mean(head_corr) * self.diversity_weight
        
        return diversity_loss
    
    def message(self, x_j, alpha):
        """Message function for aggregation"""
        # x_j: [E, heads, out_channels], alpha: [E, heads]
        alpha = alpha.unsqueeze(-1)
        return x_j * alpha
    
    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages using sum"""
        return scatter_add(inputs, index, dim=0, dim_size=dim_size)


# Utility functions for attention and message passing
def softmax(src, index, num_nodes=None):
    """
    Softmax aggregation with memory-efficient implementation
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    
    # Subtract max for numerical stability (groupwise)
    from torch_scatter import scatter_max
    max_value = scatter_max(src, index, dim=0, dim_size=num_nodes)[0]
    max_value = max_value.index_select(0, index)
    
    # Apply exp and normalization
    src = torch.exp(src - max_value)
    
    # Compute normalization constant (sum over source nodes)
    from torch_scatter import scatter_add
    norm = scatter_add(src, index, dim=0, dim_size=num_nodes)
    norm = norm.index_select(0, index)
    
    # Return normalized values
    return src / (norm + 1e-10)  # Add epsilon for numerical stability

def scatter_add(src, index, dim=-1, dim_size=None):
    """
    Efficient scatter_add implementation
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    
    # Use PyTorch's native scatter_add_ operation for efficiency
    return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)