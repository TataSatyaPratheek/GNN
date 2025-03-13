"""
Memory-efficient graph neural network architectures for process mining with 
DGL-based implementations for optimized attention mechanisms and minimal memory footprint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import numpy as np
import logging
import math
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

logger = logging.getLogger(__name__)

class BaseProcessModel(nn.Module):
    """Base class for all process mining models with unified interface"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, g):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(self, g):
        """Make predictions with a consistent interface"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(g)
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get("task_pred", next(iter(outputs.values())))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            _, predictions = torch.max(logits, dim=1)
            return predictions
    
    def get_embeddings(self, g):
        """Get embeddings from model - to be implemented by subclasses"""
        raise NotImplementedError("Embedding extraction not implemented for this model")
    
    def get_attention_weights(self, g):
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
            # Calculate input size for this layer
            if attention_type in ["basic", "positional"]:
                current_input_dim = hidden_dim * heads
            else:
                current_input_dim = hidden_dim * heads
            
            self.convs.append(self._create_conv_layer(
                current_input_dim, 
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
        
        # Set up pooling
        # In DGL, we can use readout functions for pooling
        self.pooling_func = self._set_pooling_func(pooling)
        
        # For attention pooling and combined pooling, we need additional layers
        if pooling == "attention":
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_dim * heads, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        elif pooling == "combined":
            self.pool_proj = nn.Linear(hidden_dim * heads * 3, hidden_dim * heads)
        
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
            return MemoryEfficientGATLayer(
                in_channels, 
                out_channels,
                num_heads=self.heads,
                feat_drop=self.dropout,
                residual=self.use_residual,
                sparse_attention=self.sparse_attention
            )
        elif self.attention_type == "positional":
            return PositionalGATLayer(
                in_channels, 
                out_channels,
                num_heads=self.heads,
                pos_dim=self.pos_enc_dim,
                feat_drop=self.dropout,
                residual=self.use_residual
            )
        elif self.attention_type == "diverse":
            return DiverseGATLayer(
                in_channels, 
                out_channels,
                num_heads=self.heads,
                diversity_weight=self.diversity_weight,
                feat_drop=self.dropout,
                residual=self.use_residual
            )
        elif self.attention_type == "combined":
            return CombinedGATLayer(
                in_channels, 
                out_channels,
                num_heads=self.heads,
                pos_dim=self.pos_enc_dim,
                diversity_weight=self.diversity_weight,
                feat_drop=self.dropout,
                residual=self.use_residual
            )
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
    
    def _set_pooling_func(self, pooling):
        """Set pooling function based on pooling type"""
        if pooling == "mean":
            return lambda g, h: dgl.readout_nodes(g, 'h', op='mean')
        elif pooling == "sum":
            return lambda g, h: dgl.readout_nodes(g, 'h', op='sum')
        elif pooling == "max":
            return lambda g, h: dgl.readout_nodes(g, 'h', op='max')
        elif pooling == "attention":
            # Will be handled in forward pass
            return self._attention_pooling
        elif pooling == "combined":
            # Will be handled in forward pass
            return self._combined_pooling
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    
    def _combined_pooling(self, g, h):
        """Combined pooling (concatenate mean, max, sum)"""
        mean_pool = dgl.readout_nodes(g, 'h', op='mean')
        max_pool = dgl.readout_nodes(g, 'h', op='max')
        sum_pool = dgl.readout_nodes(g, 'h', op='sum')
        combined = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
        return self.pool_proj(combined)
    
    def _attention_pooling(self, g, h):
        """Attention-based pooling"""
        # Calculate attention scores
        scores = self.pool_attention(h)
        
        # Apply softmax within each graph
        g.ndata['gate'] = scores
        g.ndata['h'] = h
        
        # Using DGL's softmax and weighted sum
        g.group_apply_edges('src', lambda edges: {'gate': torch.softmax(edges.data['gate'], dim=0)})
        g.ndata['h_weighted'] = g.ndata['h'] * g.ndata['gate']
        
        # Readout weighted feature
        return dgl.readout_nodes(g, 'h_weighted', op='sum')
    
    def _generate_positions(self, g):
        """
        Generate positional encodings for nodes in each graph
        
        Args:
            g: DGL graph or batched graph
            
        Returns:
            Position tensor [num_nodes, 2]
        """
        # Get device
        device = g.device if hasattr(g, 'device') else next(self.parameters()).device
        
        # For batched graph, calculate positions per graph
        if g.batch_size > 1:
            # Get the number of nodes per graph
            batch_num_nodes = g.batch_num_nodes()
            
            positions = []
            offset = 0
            
            for num_nodes in batch_num_nodes:
                # Calculate grid dimensions for 2D layout
                grid_size = int(math.sqrt(num_nodes)) + 1
                
                # Generate positions for this graph
                for i in range(num_nodes):
                    row = i // grid_size
                    col = i % grid_size
                    
                    # Normalize to [0,1]
                    norm_row = row / max(1, grid_size - 1)
                    norm_col = col / max(1, grid_size - 1)
                    
                    # Add position
                    positions.append([norm_row, norm_col])
                
                # Update offset
                offset += num_nodes
            
            # Create tensor of all positions
            return torch.tensor(positions, device=device)
        else:
            # Single graph
            num_nodes = g.num_nodes()
            
            # Calculate grid dimensions for 2D layout
            grid_size = int(math.sqrt(num_nodes)) + 1
            
            # Create positions tensor
            positions = torch.zeros((num_nodes, 2), device=device)
            
            # Assign positions in grid pattern
            for i in range(num_nodes):
                row = i // grid_size
                col = i % grid_size
                
                # Normalize to [0,1]
                norm_row = row / max(1, grid_size - 1)
                norm_col = col / max(1, grid_size - 1)
                
                # Assign position
                positions[i, 0] = norm_row
                positions[i, 1] = norm_col
            
            return positions
    
    def forward(self, g):
        """
        Forward pass with memory optimizations
        
        Args:
            g: DGL graph or batched graph
            
        Returns:
            Model output (format depends on configuration)
        """
        # Get node features
        h = g.ndata['feat']
        
        # Handle positional encodings
        if self.attention_type in ["positional", "combined"]:
            # Generate positions based on graph
            pos = self._generate_positions(g)
            
            # Add position embeddings to input
            pos_embedding = self.pos_encoder(pos)
            h = torch.cat([h, pos_embedding], dim=-1)
        
        # Track diversity losses and attention weights
        diversity_losses = []
        attn_weights_list = []
        
        # Memory for residual connections
        residual = None
        
        # Process through GNN layers
        for i, conv in enumerate(self.convs):
            # Apply convolutional layer with optional checkpointing
            if self.use_checkpointing and self.training and torch.torch.utils.checkpoint.is_available():
                # Use checkpointing to save memory during backprop
                if self.attention_type in ["diverse", "combined"]:
                    h_new, div_loss, attn_weights = torch.utils.checkpoint.checkpoint(
                        lambda g, h: conv(g, h, return_attention=True),
                        g, h
                    )
                    h = h_new
                    diversity_losses.append(div_loss)
                    attn_weights_list.append(attn_weights)
                else:
                    # Check if layer supports returning attention weights
                    if hasattr(conv, 'return_attention_weights'):
                        h_new, attn_weights = torch.utils.checkpoint.checkpoint(
                            lambda g, h: conv(g, h, return_attention_weights=True),
                            g, h
                        )
                        h = h_new
                        attn_weights_list.append(attn_weights)
                    else:
                        h = torch.utils.checkpoint.checkpoint(
                            lambda g, h: conv(g, h),
                            g, h
                        )
            else:
                # Standard forward pass without checkpointing
                if self.attention_type in ["diverse", "combined"]:
                    # Layers with diversity loss
                    h, div_loss, attn_weights = conv(g, h, return_attention=True)
                    diversity_losses.append(div_loss)
                    attn_weights_list.append(attn_weights)
                else:
                    # Standard or positional convolution
                    if hasattr(conv, 'return_attention_weights'):
                        h, attn_weights = conv(g, h, return_attention_weights=True)
                        attn_weights_list.append(attn_weights)
                    else:
                        h = conv(g, h)
            
            # Apply normalization if enabled
            if self.norms is not None:
                h = self.norms[i](h)
            
            # Apply activation
            h = F.elu(h)
            
            # Apply dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Apply residual connection if enabled
            if self.use_residual and i > 0 and residual is not None:
                if residual.shape == h.shape:
                    h = h + residual
            
            # Store for next residual connection
            residual = h
            
            # Aggressive memory optimization if needed
            if self.mem_efficient and i < self.num_layers - 1 and torch.cuda.is_available():
                # Free memory from intermediate computations for very large graphs
                if g.num_nodes() > 10000:  # Only for very large graphs
                    torch.cuda.empty_cache()
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights_list
        
        # Store node embeddings in graph for pooling
        g.ndata['h'] = h
        
        # Apply graph pooling
        if self.pooling_type in ["mean", "sum", "max"]:
            # Use standard readout functions
            pooled = self.pooling_func(g, h)
        elif self.pooling_type == "attention":
            # Use attention pooling
            pooled = self._attention_pooling(g, h)
        elif self.pooling_type == "combined":
            # Use combined pooling
            pooled = self._combined_pooling(g, h)
        
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
    
    def get_embeddings(self, g):
        """
        Extract node embeddings from the model
        
        Args:
            g: DGL graph or batched graph
            
        Returns:
            Node embeddings and graph IDs
        """
        self.eval()
        with torch.no_grad():
            # Get node features
            h = g.ndata['feat']
            
            # Handle positional encodings
            if self.attention_type in ["positional", "combined"]:
                pos = self._generate_positions(g)
                pos_embedding = self.pos_encoder(pos)
                h = torch.cat([h, pos_embedding], dim=-1)
            
            # Process through GNN layers
            for i, conv in enumerate(self.convs):
                # Apply convolutional layer
                if self.attention_type in ["diverse", "combined"]:
                    h, _, _ = conv(g, h, return_attention=True)
                else:
                    if hasattr(conv, 'return_attention_weights'):
                        h, _ = conv(g, h, return_attention_weights=False)
                    else:
                        h = conv(g, h)
                
                # Apply normalization if enabled
                if self.norms is not None:
                    h = self.norms[i](h)
                
                # Apply activation
                h = F.elu(h)
            
            # For batched graph, get graph IDs
            if g.batch_size > 1:
                # Create graph ID mapping (which node belongs to which graph)
                graph_ids = []
                offset = 0
                for i, num_nodes in enumerate(g.batch_num_nodes()):
                    graph_ids.extend([i] * num_nodes)
                graph_ids = torch.tensor(graph_ids, device=h.device)
            else:
                # Single graph - all nodes have same graph ID
                graph_ids = torch.zeros(g.num_nodes(), device=h.device)
            
            # Return node embeddings and graph IDs
            return h, graph_ids
    
    def get_attention_weights(self, g):
        """
        Get attention weights for interpretability
        
        Args:
            g: DGL graph or batched graph
            
        Returns:
            List of attention weights for each layer
        """
        # Run forward pass to populate attention weights
        _ = self.forward(g)
        
        # Return stored attention weights
        return self.attention_weights


class MemoryEfficientGATLayer(nn.Module):
    """
    Memory-efficient Graph Attention Layer implementation based on DGL
    with sparse attention option for processing very large graphs
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        num_heads: int = 1, 
        feat_drop: float = 0.0, 
        residual: bool = True,
        sparse_attention: bool = False,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.residual = residual
        self.sparse_attention = sparse_attention
        
        # The input feature transformation
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # The attention mechanism
        self.attn_src = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        
        # Residual connection
        if residual:
            if in_dim != out_dim * num_heads:
                self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            else:
                self.res_fc = None
        else:
            self.register_parameter('res_fc', None)
        
        # Initialize weights
        self.reset_parameters()
        
        # Message and reduce functions
        self.message_func = self._message_func
        self.reduce_func = fn.sum
        
        # Sparse attention support
        self.chunk_size = 10000 if sparse_attention else None
    
    def reset_parameters(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)
    
    def _message_func(self, edges):
        """Message function for attention mechanism"""
        # Compute attention coefficients
        src_attn = (edges.src['ft'] * self.attn_src).sum(dim=-1)
        dst_attn = (edges.dst['ft'] * self.attn_dst).sum(dim=-1)
        e = src_attn + dst_attn
        
        # Apply activation and softmax
        e = F.leaky_relu(e, negative_slope=0.2)
        
        # Return message and attention weight
        return {'m': edges.src['ft'], 'e': e}
    
    def forward(self, g, feat, return_attention_weights=False):
        """
        Forward computation
        
        Args:
            g: DGL graph or batched graph
            feat: Input node features
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features or tuple of (features, attention weights)
        """
        if self.sparse_attention and g.num_nodes() > self.chunk_size:
            return self._forward_sparse(g, feat, return_attention_weights)
        
        # Node feature transformation
        h = self.fc(feat).view(-1, self.num_heads, self.out_dim)
        
        # Apply feature dropout
        h = F.dropout(h, p=self.feat_drop, training=self.training)
        
        # Store transformed features
        g.ndata['ft'] = h
        
        # Compute attention weights
        g.apply_edges(self._message_func)
        
        # Apply softmax to attention weights (grouped by destination node)
        g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['e'])
        
        # Message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        
        # Get the updated features
        h = g.ndata['ft']
        
        # Reshape for output
        h = h.view(-1, self.num_heads * self.out_dim)
        
        # Apply residual connection if specified
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(feat)
            else:
                resval = feat
            h = h + resval
        
        # Return with attention weights if requested
        if return_attention_weights:
            return h, g.edata['a']
        else:
            return h
    
    def _forward_sparse(self, g, feat, return_attention_weights=False):
        """
        Sparse implementation of forward pass for very large graphs.
        Process the graph in chunks to reduce memory usage.
        """
        num_nodes = g.num_nodes()
        chunk_size = self.chunk_size
        
        # Node feature transformation
        h = self.fc(feat).view(-1, self.num_heads, self.out_dim)
        
        # Apply feature dropout
        h = F.dropout(h, p=self.feat_drop, training=self.training)
        
        # Store transformed features
        g.ndata['ft'] = h
        
        # Initialize output tensor
        out = torch.zeros(num_nodes, self.num_heads, self.out_dim, 
                         device=feat.device)
        
        # Store attention weights if needed
        if return_attention_weights:
            g.edata['a'] = torch.zeros(g.num_edges(), self.num_heads, 
                                      device=feat.device)
        
        # Process nodes in chunks
        for i in range(0, num_nodes, chunk_size):
            end_idx = min(i + chunk_size, num_nodes)
            chunk_nodes = list(range(i, end_idx))
            
            # Create subgraph with these nodes
            sg = g.subgraph(chunk_nodes, store_ids=True)
            
            # Compute message function for subgraph
            sg.apply_edges(self._message_func)
            
            # Apply softmax to attention weights
            sg.edata['a'] = dgl.ops.edge_softmax(sg, sg.edata['e'])
            
            # Message passing in subgraph
            sg.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            
            # Copy results back to original graph
            out[sg.ndata[dgl.NID]] = sg.ndata['ft']
            
            # Store attention weights if needed
            if return_attention_weights:
                g.edata['a'][sg.edata[dgl.EID]] = sg.edata['a']
        
        # Reshape for output
        h_out = out.view(-1, self.num_heads * self.out_dim)
        
        # Apply residual connection if specified
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(feat)
            else:
                resval = feat
            h_out = h_out + resval
        
        # Return with attention weights if requested
        if return_attention_weights:
            return h_out, g.edata['a']
        else:
            return h_out


class PositionalGATLayer(nn.Module):
    """
    Position-enhanced Graph Attention Layer optimized for memory efficiency
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        num_heads: int = 1, 
        pos_dim: int = 16,
        feat_drop: float = 0.0, 
        residual: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.feat_drop = feat_drop
        self.residual = residual
        
        # Input feature transformation
        # Adjust for positional information already in input
        content_dim = in_dim - pos_dim
        self.fc = nn.Linear(content_dim, out_dim * num_heads, bias=False)
        
        # Attention weights for content
        self.attn_src = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        
        # Attention weights for positional information
        self.pos_attn_src = nn.Parameter(torch.FloatTensor(1, num_heads, pos_dim))
        self.pos_attn_dst = nn.Parameter(torch.FloatTensor(1, num_heads, pos_dim))
        
        # Residual connection
        if residual:
            if in_dim != out_dim * num_heads:
                self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            else:
                self.res_fc = None
        else:
            self.register_parameter('res_fc', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.pos_attn_src)
        nn.init.xavier_uniform_(self.pos_attn_dst)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)
    
    def forward(self, g, feat, return_attention_weights=False):
        """
        Forward computation
        
        Args:
            g: DGL graph or batched graph
            feat: Input node features with positional encoding appended
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features or tuple of (features, attention weights)
        """
        # Split input into content and positional parts
        content = feat[:, :-self.pos_dim]
        pos = feat[:, -self.pos_dim:]
        
        # Content transformation
        h_content = self.fc(content).view(-1, self.num_heads, self.out_dim)
        
        # Apply feature dropout
        h_content = F.dropout(h_content, p=self.feat_drop, training=self.training)
        
        # Store transformed features and positional info
        g.ndata['ft'] = h_content
        g.ndata['pos'] = pos
        
        # Custom message function that computes content and positional attention
        def message_func(edges):
            # Content attention
            src_attn = (edges.src['ft'] * self.attn_src).sum(dim=-1)
            dst_attn = (edges.dst['ft'] * self.attn_dst).sum(dim=-1)
            content_e = src_attn + dst_attn
            
            # Positional attention
            src_pos_attn = (edges.src['pos'].unsqueeze(1) * self.pos_attn_src).sum(dim=-1)
            dst_pos_attn = (edges.dst['pos'].unsqueeze(1) * self.pos_attn_dst).sum(dim=-1)
            pos_e = src_pos_attn + dst_pos_attn
            
            # Combine content and positional attention
            e = content_e + pos_e
            e = F.leaky_relu(e, negative_slope=0.2)
            
            return {'m': edges.src['ft'], 'e': e}
        
        # Apply message function to edges
        g.apply_edges(message_func)
        
        # Apply softmax to attention weights
        g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['e'])
        
        # Message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        
        # Get the updated features
        h = g.ndata['ft']
        
        # Reshape for output
        h = h.view(-1, self.num_heads * self.out_dim)
        
        # Apply residual connection if specified
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(feat)
            else:
                resval = feat
            h = h + resval
        
        # Return with attention weights if requested
        if return_attention_weights:
            return h, g.edata['a']
        else:
            return h


class DiverseGATLayer(nn.Module):
    """
    GAT layer with attention diversity loss to avoid attention collapse
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        num_heads: int = 4, 
        diversity_weight: float = 0.1, 
        feat_drop: float = 0.0, 
        residual: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.diversity_weight = diversity_weight
        self.feat_drop = feat_drop
        self.residual = residual
        
        # Input feature transformation
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # Attention weights
        self.attn_src = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        
        # Residual connection
        if residual:
            if in_dim != out_dim * num_heads:
                self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            else:
                self.res_fc = None
        else:
            self.register_parameter('res_fc', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)
    
    def forward(self, g, feat, return_attention=False):
        """
        Forward pass with improved diversity loss
        
        Args:
            g: DGL graph or batched graph
            feat: Input node features
            return_attention: Whether to return attention weights and diversity loss
            
        Returns:
            If return_attention=False:
                Updated node features
            If return_attention=True:
                Tuple of (node features, diversity_loss, attention_weights)
        """
        # Feature transformation
        h = self.fc(feat).view(-1, self.num_heads, self.out_dim)
        
        # Apply feature dropout
        h = F.dropout(h, p=self.feat_drop, training=self.training)
        
        # Store transformed features
        g.ndata['ft'] = h
        
        # Compute attention scores
        def message_func(edges):
            src_attn = (edges.src['ft'] * self.attn_src).sum(dim=-1)
            dst_attn = (edges.dst['ft'] * self.attn_dst).sum(dim=-1)
            e = src_attn + dst_attn
            e = F.leaky_relu(e, negative_slope=0.2)
            return {'e': e}
        
        g.apply_edges(message_func)
        
        # Apply softmax to attention weights
        g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['e'])
        
        # Compute diversity loss if in training mode
        if self.training or return_attention:
            diversity_loss = self._calculate_diversity_loss(g)
        else:
            diversity_loss = torch.tensor(0.0, device=feat.device)
        
        # Message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        
        # Get the updated features
        h = g.ndata['ft']
        
        # Reshape for output
        h = h.view(-1, self.num_heads * self.out_dim)
        
        # Apply residual connection if specified
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(feat)
            else:
                resval = feat
            h = h + resval
        
        # Return with diversity loss and attention weights if requested
        if return_attention:
            return h, diversity_loss, g.edata['a']
        else:
            return h, diversity_loss
    
    def _calculate_diversity_loss(self, g):
        """
        Calculate diversity loss to encourage different attention patterns
        
        Args:
            g: DGL graph with attention weights in edata['a']
            
        Returns:
            Diversity loss tensor
        """
        if self.num_heads <= 1:
            return torch.tensor(0.0, device=g.device)
        
        # Step 1: Compute head correlations
        # Get attention weights: [E, num_heads]
        attn = g.edata['a']
        
        # Calculate pairwise cosine similarity between attention heads
        head_sim = torch.zeros(self.num_heads, self.num_heads, device=g.device)
        
        for i in range(self.num_heads):
            for j in range(i+1, self.num_heads):
                # Extract attention weights for each head
                a_i = attn[:, i]
                a_j = attn[:, j]
                
                # Normalize for numerical stability
                a_i = F.normalize(a_i, p=2, dim=0)
                a_j = F.normalize(a_j, p=2, dim=0)
                
                # Compute cosine similarity
                sim = torch.sum(a_i * a_j)
                
                # Store similarity (symmetrically)
                head_sim[i, j] = sim
                head_sim[j, i] = sim
        
        # Step 2: Calculate global head similarity
        # (exclude diagonal elements)
        mask = ~torch.eye(self.num_heads, dtype=torch.bool, device=g.device)
        global_similarity = head_sim[mask].mean()
        
        # Step 3: Calculate node-level diversity
        node_diversity = torch.zeros(1, device=g.device)
        
        # For each node, calculate entropy of attention distribution
        for ntype in g.ntypes:
            # Get nodes of this type
            nodes = g.nodes(ntype)
            
            # Skip if no nodes
            if len(nodes) == 0:
                continue
            
            # Get incoming edges for these nodes
            in_edges = g.in_edges(nodes, form='eid')
            
            if len(in_edges) == 0:
                continue
            
            # Get attention weights for these edges
            edge_attn = g.edata['a'][in_edges]
            
            # Calculate entropy for each head's attention distribution
            entropy = torch.zeros(self.num_heads, device=g.device)
            
            for h in range(self.num_heads):
                # Get normalized attention weights
                head_attn = F.normalize(edge_attn[:, h], p=1, dim=0)
                
                # Calculate entropy (avoid log(0) by adding epsilon)
                epsilon = 1e-10
                head_entropy = -torch.sum(head_attn * torch.log(head_attn + epsilon))
                entropy[h] = head_entropy
            
            # Node diversity is the negative of average entropy
            # (negative because we want to maximize entropy = maximize diversity)
            node_diversity = -entropy.mean()
        
        # Combine global and node-level diversities
        # Higher similarity and lower entropy (higher node_diversity) mean less diversity
        diversity_loss = (global_similarity * 0.7 + node_diversity * 0.3) * self.diversity_weight
        
        return diversity_loss


class CombinedGATLayer(nn.Module):
    """
    Combined GAT layer with position-aware diverse attention
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        num_heads: int = 4, 
        pos_dim: int = 16,
        diversity_weight: float = 0.1, 
        pos_weight: float = 0.5,
        feat_drop: float = 0.0, 
        residual: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.pos_dim = pos_dim
        self.diversity_weight = diversity_weight
        self.pos_weight = pos_weight
        self.feat_drop = feat_drop
        self.residual = residual
        
        # Content features dimension (excluding position)
        content_dim = in_dim - pos_dim
        
        # Main transformation for content features
        self.content_fc = nn.Linear(content_dim, out_dim * num_heads, bias=False)
        
        # Transformation for positional features
        self.pos_fc = nn.Linear(pos_dim, out_dim // 4 * num_heads, bias=False)
        
        # Attention weights for content
        self.attn_src = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        
        # Attention weights for position
        self.pos_attn_src = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim // 4))
        self.pos_attn_dst = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim // 4))
        
        # Weights to combine attention components
        self.att_combination = nn.Parameter(torch.FloatTensor(num_heads, 2))
        
        # Residual connection
        if residual:
            if in_dim != out_dim * num_heads:
                self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            else:
                self.res_fc = None
        else:
            self.register_parameter('res_fc', None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.content_fc.weight)
        nn.init.xavier_uniform_(self.pos_fc.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.xavier_uniform_(self.pos_attn_src)
        nn.init.xavier_uniform_(self.pos_attn_dst)
        
        # Initialize combination weights to balance content and position
        nn.init.constant_(self.att_combination, 0.5)
        
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)
    
    def forward(self, g, feat, return_attention=False):
        """
        Forward pass with combined positional and diverse attention
        
        Args:
            g: DGL graph or batched graph
            feat: Input node features including positional features
            return_attention: Whether to return attention weights and diversity loss
            
        Returns:
            If return_attention=False:
                Updated node features
            If return_attention=True:
                Tuple of (node features, diversity_loss, attention_weights)
        """
        # Split features into content and position
        content = feat[:, :-self.pos_dim]
        pos = feat[:, -self.pos_dim:]
        
        # Transform features
        h_content = self.content_fc(content).view(-1, self.num_heads, self.out_dim)
        h_pos = self.pos_fc(pos).view(-1, self.num_heads, self.out_dim // 4)
        
        # Apply feature dropout
        h_content = F.dropout(h_content, p=self.feat_drop, training=self.training)
        h_pos = F.dropout(h_pos, p=self.feat_drop, training=self.training)
        
        # Store transformed features
        g.ndata['content'] = h_content
        g.ndata['pos'] = h_pos
        
        # Custom message function that computes combined attention
        def message_func(edges):
            # Content attention
            src_attn = (edges.src['content'] * self.attn_src).sum(dim=-1)
            dst_attn = (edges.dst['content'] * self.attn_dst).sum(dim=-1)
            content_e = src_attn + dst_attn
            
            # Positional attention
            src_pos_attn = (edges.src['pos'] * self.pos_attn_src).sum(dim=-1)
            dst_pos_attn = (edges.dst['pos'] * self.pos_attn_dst).sum(dim=-1)
            pos_e = src_pos_attn + dst_pos_attn
            
            # Combine attention components with learned weights
            combined_e = (
                content_e * self.att_combination[:, 0].unsqueeze(0) + 
                pos_e * self.att_combination[:, 1].unsqueeze(0)
            )
            
            combined_e = F.leaky_relu(combined_e, negative_slope=0.2)
            
            return {'m': edges.src['content'], 'e': combined_e}
        
        # Apply message function to edges
        g.apply_edges(message_func)
        
        # Apply softmax to attention weights
        g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['e'])
        
        # Compute diversity loss if in training mode
        if self.training or return_attention:
            diversity_loss = self._calculate_diversity_loss(g)
        else:
            diversity_loss = torch.tensor(0.0, device=feat.device)
        
        # Message passing
        g.update_all(fn.u_mul_e('content', 'a', 'm'), fn.sum('m', 'ft'))
        
        # Get the updated features
        h = g.ndata['ft']
        
        # Reshape for output
        h = h.view(-1, self.num_heads * self.out_dim)
        
        # Apply residual connection if specified
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(feat)
            else:
                resval = feat
            h = h + resval
        
        # Return with diversity loss and attention weights if requested
        if return_attention:
            return h, diversity_loss, g.edata['a']
        else:
            return h, diversity_loss
    
    def _calculate_diversity_loss(self, g):
        """
        Calculate diversity loss with improved implementation
        
        Args:
            g: DGL graph with attention weights in edata['a']
            
        Returns:
            Diversity loss tensor
        """
        if self.num_heads <= 1:
            return torch.tensor(0.0, device=g.device)
        
        # Step 1: Compute head correlations
        # Get attention weights: [E, num_heads]
        attn = g.edata['a']
        
        # Calculate pairwise cosine similarity between attention heads
        head_sim = torch.zeros(self.num_heads, self.num_heads, device=g.device)
        
        for i in range(self.num_heads):
            for j in range(i+1, self.num_heads):
                # Extract attention weights for each head
                a_i = attn[:, i]
                a_j = attn[:, j]
                
                # Normalize for numerical stability
                a_i = F.normalize(a_i, p=2, dim=0)
                a_j = F.normalize(a_j, p=2, dim=0)
                
                # Compute cosine similarity
                sim = torch.sum(a_i * a_j)
                
                # Store similarity (symmetrically)
                head_sim[i, j] = sim
                head_sim[j, i] = sim
        
        # Step 2: Calculate global head similarity
        # (exclude diagonal elements)
        mask = ~torch.eye(self.num_heads, dtype=torch.bool, device=g.device)
        global_similarity = head_sim[mask].mean()
        
        # Step 3: Calculate node-level diversity
        # For each node, calculate entropy of attention distribution
        entropy_per_node = []
        
        # Get source nodes for each edge
        src_nodes = g.edges()[0]
        
        # Convert to numpy for faster computation if tensor is large
        if len(src_nodes) > 10000:
            src_nodes_np = src_nodes.cpu().numpy()
            unique_nodes, inverse_indices = np.unique(src_nodes_np, return_inverse=True)
            unique_nodes = torch.tensor(unique_nodes, device=g.device)
        else:
            unique_nodes, inverse_indices = torch.unique(src_nodes, return_inverse=True)
        
        # For each unique source node
        for node_idx in range(len(unique_nodes)):
            # Find edges with this source node
            edge_mask = (inverse_indices == node_idx)
            
            # Get attention for these edges
            node_attn = attn[edge_mask]
            
            # Skip if no edges
            if len(node_attn) <= 1:
                continue
            
            # Calculate entropy for each head
            for h in range(self.num_heads):
                head_attn = node_attn[:, h]
                
                # Normalize
                head_attn = F.normalize(head_attn, p=1, dim=0)
                
                # Calculate entropy (avoid log(0) by adding epsilon)
                epsilon = 1e-10
                entropy = -torch.sum(head_attn * torch.log(head_attn + epsilon))
                entropy_per_node.append(entropy.item())
        
        # Calculate average entropy (higher is better)
        avg_entropy = torch.tensor(
            sum(entropy_per_node) / len(entropy_per_node) if entropy_per_node else 0.0,
            device=g.device
        )
        
        # Combine metrics - balance head similarity and node diversity
        # Higher similarity and lower entropy mean less diversity
        diversity_loss = (global_similarity * 0.7 - avg_entropy * 0.3) * self.diversity_weight
        
        return diversity_loss
    
class ProcessLoss(nn.Module):
    """
    Multi-objective loss function for process mining with DGL compatibility
    """
    def __init__(self, task_weight=0.5, time_weight=0.3, structure_weight=0.2):
        """
        Initialize process loss with component weights
        
        Args:
            task_weight: Weight for task prediction loss
            time_weight: Weight for time prediction loss
            structure_weight: Weight for structural loss
        """
        super().__init__()
        self.task_weight = task_weight
        self.time_weight = time_weight
        self.structure_weight = structure_weight
        self.task_loss = nn.CrossEntropyLoss()
        self.time_loss = nn.MSELoss()
    
    def forward(self, task_pred, task_target, g=None, time_pred=None, time_target=None, 
                structure_info=None):
        """
        Forward pass with multiple loss components
        
        Args:
            task_pred: Task prediction logits
            task_target: Task ground truth
            g: DGL graph or batched graph (for structural loss)
            time_pred: Time prediction (optional)
            time_target: Time ground truth (optional)
            structure_info: Additional structural information (optional)
            
        Returns:
            Tuple of (combined_loss, component_dict)
        """
        # Task prediction loss
        task_loss = self.task_loss(task_pred, task_target)
        
        # Initialize other losses
        time_loss = torch.tensor(0.0, device=task_pred.device)
        structure_loss = torch.tensor(0.0, device=task_pred.device)
        
        # Cycle time prediction loss if available
        if time_pred is not None and time_target is not None:
            time_loss = self.time_loss(time_pred, time_target)
        
        # Structural loss if graph and embeddings available
        if g is not None and 'h' in g.ndata and structure_info is not None:
            # Extract critical path nodes if provided
            critical_path = structure_info.get('critical_path', [])
            if critical_path:
                # Get node embeddings for critical path nodes
                if g.batch_size > 1:
                    # For batched graph, we need to map the critical path nodes
                    # to the correct positions in the batched graph
                    batch_indices = structure_info.get('batch_indices', [])
                    if batch_indices:
                        embeddings = g.ndata['h']
                        cp_embeddings = []
                        for node_idx, batch_idx in zip(critical_path, batch_indices):
                            # Calculate offset in the batched graph
                            offset = 0
                            for i in range(batch_idx):
                                offset += g.batch_num_nodes()[i]
                            cp_embeddings.append(embeddings[offset + node_idx])
                        
                        if cp_embeddings:
                            cp_embeddings = torch.stack(cp_embeddings)
                            # Minimize distance between consecutive nodes in critical path
                            diffs = cp_embeddings[1:] - cp_embeddings[:-1]
                            structure_loss = torch.mean(torch.norm(diffs, dim=1))
                else:
                    # For single graph, just index directly
                    embeddings = g.ndata['h']
                    cp_embeddings = embeddings[critical_path]
                    # Minimize distance between consecutive nodes in critical path
                    diffs = cp_embeddings[1:] - cp_embeddings[:-1]
                    structure_loss = torch.mean(torch.norm(diffs, dim=1))
        
        # Combine losses with weights
        combined_loss = (
            self.task_weight * task_loss +
            self.time_weight * time_loss +
            self.structure_weight * structure_loss
        )
        
        return combined_loss, {
            'task_loss': task_loss.item(),
            'time_loss': time_loss.item(),
            'structure_loss': structure_loss.item(),
            'combined_loss': combined_loss.item()
        }

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for graph neural networks
    Implements the position-aware features described in the improvement plan
    """
    def __init__(self, input_dim, pos_dim=16, max_len=1000):
        """
        Initialize positional encoding module
        
        Args:
            input_dim: Dimension of input features
            pos_dim: Dimension of positional encoding
            max_len: Maximum sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.max_len = max_len
        
        # Learnable position embedding
        self.pos_embedding = nn.Embedding(max_len, pos_dim)
        
        # Projection to combine with input
        self.pos_projection = nn.Linear(input_dim + pos_dim, input_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.pos_projection.weight)
        nn.init.zeros_(self.pos_projection.bias)
    
    def forward(self, x, positions=None):
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, seq_len, input_dim] or [num_nodes, input_dim]
            positions: Optional explicit positions. If None, uses node indices.
            
        Returns:
            Position-enhanced features [batch_size, seq_len, input_dim] or [num_nodes, input_dim]
        """
        if x.dim() == 2:
            # [num_nodes, input_dim]
            num_nodes = x.size(0)
            if positions is None:
                # Use node indices as positions
                positions = torch.arange(num_nodes, device=x.device)
            
            # Clamp positions to max_len
            positions = torch.clamp(positions, 0, self.max_len - 1)
            
            # Get position embeddings [num_nodes, pos_dim]
            pos_emb = self.pos_embedding(positions)
            
            # Concatenate with input features [num_nodes, input_dim + pos_dim]
            enhanced = torch.cat([x, pos_emb], dim=-1)
            
            # Project back to original dimension [num_nodes, input_dim]
            return self.pos_projection(enhanced)
            
        elif x.dim() == 3:
            # [batch_size, seq_len, input_dim]
            batch_size, seq_len = x.size(0), x.size(1)
            if positions is None:
                # Use sequence indices as positions
                positions = torch.arange(seq_len, device=x.device).expand(batch_size, -1)
            
            # Clamp positions to max_len
            positions = torch.clamp(positions, 0, self.max_len - 1)
            
            # Get position embeddings [batch_size, seq_len, pos_dim]
            pos_emb = self.pos_embedding(positions)
            
            # Concatenate with input features [batch_size, seq_len, input_dim + pos_dim]
            enhanced = torch.cat([x, pos_emb], dim=-1)
            
            # Project back to original dimension [batch_size, seq_len, input_dim]
            return self.pos_projection(enhanced)
        
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


# Update processmine/models/gnn/architectures.py to ensure ExpressiveGATConv properly uses DGL

class ExpressiveGATConv(nn.Module):
    """
    More expressive GAT convolution with DGL-optimized implementation
    """
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        num_heads=4, 
        feat_drop=0.0, 
        attn_drop=0.0, 
        negative_slope=0.2,
        residual=True,
        allow_zero_in_degree=False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.allow_zero_in_degree = allow_zero_in_degree
        
        # Multi-head attention
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        self.attn_e = nn.Parameter(torch.FloatTensor(1, num_heads, out_dim))
        
        # Additional expressive components from improvement plan
        self.gate = nn.Linear(in_dim, num_heads)
        self.feat_proj = nn.Linear(in_dim, out_dim * num_heads)
        
        # Residual connection
        if residual:
            if in_dim != out_dim * num_heads:
                self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        # Dropout
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.feat_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        nn.init.xavier_normal_(self.gate.weight, gain=gain)
        
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
    
    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        """
        Forward computation with DGL
        
        Args:
            graph: DGL graph object
            feat: Node features
            edge_feat: Edge features (optional)
            get_attention: Whether to return attention weights
            
        Returns:
            Updated node features or tuple of (features, attention weights)
        """
        with graph.local_scope():
            # Feature dropout
            h_src = h_dst = self.feat_drop(feat)
            
            # Feature transformation
            feat_src = feat_dst = self.fc(h_src).view(-1, self.num_heads, self.out_dim)
            
            # Additional feature projection (more expressive)
            feat_proj = self.feat_proj(h_src).view(-1, self.num_heads, self.out_dim)
            
            # Gating mechanism for feature importance
            gate = torch.sigmoid(self.gate(h_src)).view(-1, self.num_heads, 1)
            feat_proj = feat_proj * gate
            
            # Prepare for attention
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            
            # Add to graph
            graph.srcdata.update({'ft': feat_src, 'el': el, 'proj': feat_proj})
            graph.dstdata.update({'er': er})
            
            # Handle edge features if provided
            if edge_feat is not None:
                if edge_feat.dim() != 3:
                    # Reshape edge features to [num_edges, num_heads, out_dim]
                    edge_feat = edge_feat.view(-1, 1, 1).expand(-1, self.num_heads, 1)
                graph.edata.update({'ee': edge_feat})
                
                # Define custom message function with edge features
                def message_func(edges):
                    # Combine source, destination, and edge attention
                    ee = (edges.data['ee'] * self.attn_e).sum(dim=-1).unsqueeze(-1)
                    a = self.leaky_relu(edges.src['el'] + edges.dst['er'] + ee)
                    a = self.attn_drop(a)
                    return {'a': a, 'm': edges.src['ft'] + edges.src['proj']}
            else:
                # Standard message function without edge features
                def message_func(edges):
                    a = self.leaky_relu(edges.src['el'] + edges.dst['er'])
                    a = self.attn_drop(a)
                    return {'a': a, 'm': edges.src['ft'] + edges.src['proj']}
            
            # Apply edge softmax using DGL's optimized function
            graph.apply_edges(message_func)
            graph.edata['a'] = dgl.nn.functional.edge_softmax(graph, graph.edata['a'])
            
            # Define reduce function
            def reduce_func(nodes):
                alpha = nodes.mailbox['a']
                h = torch.sum(alpha * nodes.mailbox['m'], dim=1)
                return {'h': h}
            
            # Update all nodes
            graph.update_all(fn.src_mul_edge('proj', 'a', 'm'), fn.sum('m', 'h'))
            
            # Get updated features
            h = graph.dstdata['h']
            
            # Apply residual connection if specified
            if self.residual:
                h = h + self.res_fc(feat)
            
            # Return attention weights if requested
            if get_attention:
                return h, graph.edata['a']
            else:
                return h