"""
Unified and optimized GNN architectures for process mining with memory-efficient components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import logging
from typing import Dict, Any, Optional, Union, Tuple, List

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
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
    
    def parameter_count(self):
        """Get parameter count for the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def memory_usage(self):
        """Estimate model memory usage in MB"""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        total_mb = (param_bytes + buffer_bytes) / (1024 * 1024)
        return {
            'parameters_mb': param_bytes / (1024 * 1024),
            'buffers_mb': buffer_bytes / (1024 * 1024),
            'total_mb': total_mb
        }

class OptimizedGNN(BaseModel):
    """
    Memory-efficient GNN architecture with unified interface for process mining
    Supports various attention mechanisms and pooling strategies
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
        use_residual: bool = True,
        mem_efficient: bool = True
    ):
        """
        Initialize optimized GNN
        
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
            pooling: Pooling method ("mean", "sum", "max", "combined")
            predict_time: Whether to predict time in addition to task
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            mem_efficient: Whether to use memory-efficient implementation
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
        self.use_residual = use_residual
        self.mem_efficient = mem_efficient
        
        # Track attention weights for interpretability
        self.attention_weights = None
        
        # Import appropriate layers based on attention type
        if attention_type == "basic":
            from torch_geometric.nn import GATConv
            
            # Create layers
            self.convs = nn.ModuleList()
            
            # Input layer
            self.convs.append(GATConv(
                input_dim, hidden_dim,
                heads=heads,
                dropout=dropout,
                add_self_loops=True
            ))
            
            # Hidden layers with possible residual connections
            for i in range(num_layers - 1):
                if use_residual and i > 0:
                    # Residual connection - input is previous layer's output + original
                    self.convs.append(GATConv(
                        hidden_dim * heads, hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=True
                    ))
                else:
                    # Standard connection
                    self.convs.append(GATConv(
                        hidden_dim * heads, hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        add_self_loops=True
                    ))
                
        elif attention_type == "positional":
            # Use our custom positional GAT implementation
            self.pos_encoder = nn.Linear(2, pos_enc_dim)
            input_with_pos = input_dim + pos_enc_dim
            
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    # First layer takes input_dim + pos_enc_dim
                    self.convs.append(PositionalGATConv(
                        input_with_pos, hidden_dim,
                        heads=heads,
                        pos_dim=pos_enc_dim,
                        dropout=dropout
                    ))
                else:
                    # Subsequent layers take hidden_dim * heads
                    self.convs.append(PositionalGATConv(
                        hidden_dim * heads, hidden_dim,
                        heads=heads,
                        pos_dim=pos_enc_dim,
                        dropout=dropout
                    ))
                
        elif attention_type == "diverse":
            # Use our custom diverse attention implementation 
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    # First layer takes input_dim
                    self.convs.append(DiverseGATConv(
                        input_dim, hidden_dim,
                        heads=heads,
                        diversity_weight=diversity_weight,
                        dropout=dropout
                    ))
                else:
                    # Subsequent layers take hidden_dim * heads
                    self.convs.append(DiverseGATConv(
                        hidden_dim * heads, hidden_dim,
                        heads=heads,
                        diversity_weight=diversity_weight,
                        dropout=dropout
                    ))
                
        elif attention_type == "combined":
            # Combined attention with positional encoding and diversity
            self.pos_encoder = nn.Linear(2, pos_enc_dim)
            input_with_pos = input_dim + pos_enc_dim
            
            self.convs = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    # First layer takes input_dim + pos_enc_dim
                    self.convs.append(CombinedGATConv(
                        input_with_pos, hidden_dim,
                        heads=heads,
                        pos_dim=pos_enc_dim,
                        diversity_weight=diversity_weight,
                        dropout=dropout
                    ))
                else:
                    # Subsequent layers take hidden_dim * heads
                    self.convs.append(CombinedGATConv(
                        hidden_dim * heads, hidden_dim,
                        heads=heads,
                        pos_dim=pos_enc_dim,
                        diversity_weight=diversity_weight,
                        dropout=dropout
                    ))
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim * heads) for _ in range(num_layers)
            ])
        else:
            self.batch_norms = None
        
        # Set pooling function
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "combined":
            # Combined pooling (concatenate mean, max, sum)
            self.pool = self._combined_pooling
            # Adjust output projection to account for combined pooling
            combined_dim = hidden_dim * heads * 3
            self.pool_proj = nn.Linear(combined_dim, hidden_dim * heads)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Output layers
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
    
    def _combined_pooling(self, x, batch):
        """Combined pooling (concatenate mean, max, sum)"""
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        combined = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
        return self.pool_proj(combined)
    
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
            
            # Calculate grid dimensions
            grid_size = int(count_val ** 0.5) + 1
            
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
        Forward pass through the GNN
        
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
        
        # Track diversity losses if needed
        diversity_losses = []
        
        # Memory for residual connections
        residual = None
        
        # Clear stored attention weights
        self.attention_weights = []
        
        # Process through GNN layers
        for i, conv in enumerate(self.convs):
            # Apply convolution based on attention type
            if self.attention_type == "diverse" or self.attention_type == "combined":
                # These return (output, diversity_loss, attention)
                layer_out, div_loss, att_weights = conv(x, edge_index, return_attention=True)
                x = layer_out
                diversity_losses.append(div_loss)
                
                # Store attention weights
                self.attention_weights.append(att_weights)
            else:
                # Standard or positional convolution
                if hasattr(conv, 'return_attention_weights'):
                    # GAT with attention weights
                    x, att_weights = conv(x, edge_index, return_attention_weights=True)
                    self.attention_weights.append(att_weights)
                else:
                    # Standard convolution
                    x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
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
            if self.mem_efficient and i < self.num_layers - 1:
                # Free memory from intermediate computations
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
                # Apply convolution
                if self.attention_type == "diverse" or self.attention_type == "combined":
                    x, _, _ = conv(x, edge_index, return_attention=True)
                else:
                    if hasattr(conv, 'return_attention_weights'):
                        x, _ = conv(x, edge_index, return_attention_weights=False)
                    else:
                        x = conv(x, edge_index)
                
                # Apply batch normalization if enabled
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                
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


class PositionalGATConv(nn.Module):
    """
    Position-enhanced Graph Attention Layer with efficient implementation
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
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Main transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention weights
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
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.pos_att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention_weights=False):
        """Forward pass with optional attention weights return"""
        from torch_geometric.utils import remove_self_loops, add_self_loops
        from torch_geometric.utils import softmax
        
        # Remove and add self-loops
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)
        
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Propagate
        # Extract node features for source and target nodes
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Concatenate source and target features
        alpha = torch.cat([x_i, x_j], dim=-1)
        
        # Apply attention
        alpha = (alpha * self.att).sum(dim=-1)
        
        # Add positional component if available
        if x.size(1) > self.out_channels * self.heads:
            # Extract positional features
            pos_dim = (x.size(1) - self.out_channels * self.heads) // self.heads
            pos_i, pos_j = x[row, :, -pos_dim:], x[col, :, -pos_dim:]
            
            # Compute positional attention
            pos_alpha = torch.cat([pos_i, pos_j], dim=-1)
            pos_alpha = (pos_alpha * self.pos_att).sum(dim=-1)
            
            # Combine with feature attention
            alpha = alpha + pos_alpha
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted sum
        out = x_j * alpha.view(-1, self.heads, 1)
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))
        
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


class DiverseGATConv(nn.Module):
    """
    GAT layer with attention diversity to avoid attention collapse
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
        super().__init__()
        
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
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention=False):
        """Forward pass with diversity loss"""
        from torch_geometric.utils import remove_self_loops, add_self_loops
        from torch_geometric.utils import softmax
        
        # Remove and add self-loops
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)
        
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Propagate
        # Extract node features for source and target nodes
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Concatenate source and target features
        alpha = torch.cat([x_i, x_j], dim=-1)
        
        # Apply attention
        alpha = (alpha * self.att).sum(dim=-1)
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # Calculate diversity loss (covariance across heads)
        head_corr = torch.zeros(self.heads, self.heads, device=alpha.device)
        for h1 in range(self.heads):
            for h2 in range(h1+1, self.heads):
                # Calculate correlation between attention heads
                corr = F.cosine_similarity(
                    alpha[:, h1].unsqueeze(0), 
                    alpha[:, h2].unsqueeze(0)
                )
                head_corr[h1, h2] = corr
                head_corr[h2, h1] = corr
        
        # Get average correlation - positive value indicates high correlation
        diversity_loss = torch.mean(head_corr) * self.diversity_weight
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted sum
        out = x_j * alpha.view(-1, self.heads, 1)
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))
        
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


class CombinedGATConv(nn.Module):
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
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.pos_dim = pos_dim
        self.diversity_weight = diversity_weight
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Main transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention weights
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
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.pos_att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, return_attention=False):
        """Forward pass with diversity loss and position-aware attention"""
        from torch_geometric.utils import remove_self_loops, add_self_loops
        from torch_geometric.utils import softmax
        
        # Remove and add self-loops
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)
        
        # Split features into content and position parts
        pos_features = x[:, -self.pos_dim:]
        content_features = x[:, :-self.pos_dim]
        
        # Apply content transformation
        content = self.lin(content_features).view(-1, self.heads, self.out_channels)
        
        # Propagate
        # Extract node features for source and target nodes
        row, col = edge_index
        content_i, content_j = content[row], content[col]
        pos_i, pos_j = pos_features[row], pos_features[col]
        
        # Compute content attention
        alpha_content = torch.cat([content_i, content_j], dim=-1)
        alpha_content = (alpha_content * self.att).sum(dim=-1)
        
        # Compute positional attention
        alpha_pos = torch.cat([pos_i.unsqueeze(1).repeat(1, self.heads, 1), 
                              pos_j.unsqueeze(1).repeat(1, self.heads, 1)], dim=-1)
        alpha_pos = (alpha_pos * self.pos_att).sum(dim=-1)
        
        # Combine attentions
        alpha = alpha_content + alpha_pos
        
        # Apply LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention weights
        alpha = softmax(alpha, row, num_nodes=content.size(0))
        
        # Calculate diversity loss (covariance across heads)
        head_corr = torch.zeros(self.heads, self.heads, device=alpha.device)
        for h1 in range(self.heads):
            for h2 in range(h1+1, self.heads):
                # Calculate correlation between attention heads
                corr = F.cosine_similarity(
                    alpha[:, h1].unsqueeze(0), 
                    alpha[:, h2].unsqueeze(0)
                )
                head_corr[h1, h2] = corr
                head_corr[h2, h1] = corr
        
        # Get average correlation - positive value indicates high correlation
        diversity_loss = torch.mean(head_corr) * self.diversity_weight
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply weighted sum
        out = content_j * alpha.view(-1, self.heads, 1)
        out = scatter_add(out, row, dim=0, dim_size=content.size(0))
        
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

# Utility function for scatter operations
def scatter_add(src, index, dim=-1, dim_size=None):
    """
    Efficient scatter add implementation
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    
    # Use PyTorch's native scatter operation
    return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)