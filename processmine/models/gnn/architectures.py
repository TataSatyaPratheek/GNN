"""
Unified GNN architecture for process mining
Supports different attention mechanisms and enhancements
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import logging
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ConfigurableGNN(nn.Module):
    """
    Unified configurable GNN for process mining
    Supports different attention mechanisms and enhancements
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
        pos_dim: int = 16,
        diversity_weight: float = 0.1,
        pooling: str = "mean",
        predict_time: bool = False,
        use_batch_norm: bool = True
    ):
        """
        Initialize configurable GNN
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            num_layers: Number of GNN layers
            heads: Number of attention heads
            dropout: Dropout probability
            attention_type: Type of attention ("basic", "positional", "diverse", "combined")
            pos_dim: Positional encoding dimension
            diversity_weight: Weight for diversity loss
            pooling: Pooling method ("mean", "sum", "max")
            predict_time: Whether to predict time in addition to task
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.attention_type = attention_type
        self.pos_dim = pos_dim
        self.diversity_weight = diversity_weight
        self.predict_time = predict_time
        
        # Import appropriate layer based on attention type
        if attention_type == "basic":
            from torch_geometric.nn import GATConv
            
            # Create layers
            self.convs = nn.ModuleList()
            
            # Input layer
            self.convs.append(GATConv(
                input_dim, hidden_dim,
                heads=heads,
                dropout=dropout
            ))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(
                    hidden_dim * heads, hidden_dim,
                    heads=heads,
                    dropout=dropout
                ))
                
        elif attention_type == "positional":
            from .layers import PositionalGATConv
            
            # Position encoder
            self.pos_encoder = nn.Linear(2, pos_dim)
            input_with_pos = input_dim + pos_dim
            
            # Create layers
            self.convs = nn.ModuleList()
            
            # Input layer
            self.convs.append(PositionalGATConv(
                input_with_pos, hidden_dim,
                heads=heads,
                pos_dim=pos_dim,
                dropout=dropout
            ))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(PositionalGATConv(
                    hidden_dim * heads, hidden_dim,
                    heads=heads,
                    pos_dim=pos_dim,
                    dropout=dropout
                ))
                
        elif attention_type == "diverse":
            from .layers import DiverseGATConv
            
            # Create layers
            self.convs = nn.ModuleList()
            
            # Input layer
            self.convs.append(DiverseGATConv(
                input_dim, hidden_dim,
                heads=heads,
                diversity_weight=diversity_weight,
                dropout=dropout
            ))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(DiverseGATConv(
                    hidden_dim * heads, hidden_dim,
                    heads=heads,
                    diversity_weight=diversity_weight,
                    dropout=dropout
                ))
                
        elif attention_type == "combined":
            from .layers import CombinedGATConv
            
            # Position encoder
            self.pos_encoder = nn.Linear(2, pos_dim)
            input_with_pos = input_dim + pos_dim
            
            # Create layers
            self.convs = nn.ModuleList()
            
            # Input layer
            self.convs.append(CombinedGATConv(
                input_with_pos, hidden_dim,
                heads=heads,
                pos_dim=pos_dim,
                diversity_weight=diversity_weight,
                dropout=dropout
            ))
            
            # Hidden layers
            for _ in range(num_layers - 1):
                self.convs.append(CombinedGATConv(
                    hidden_dim * heads, hidden_dim,
                    heads=heads,
                    pos_dim=pos_dim,
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
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.time_pred = None
    
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
        
        # Process through GNN layers
        for i, conv in enumerate(self.convs):
            # Apply convolution based on attention type
            if self.attention_type == "diverse" or self.attention_type == "combined":
                # These return (output, diversity_loss)
                x, div_loss = conv(x, edge_index)
                diversity_losses.append(div_loss)
            else:
                # Standard convolution
                x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
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
        
        return outputs
    
    def _generate_positions(self, batch):
        """
        Generate normalized positions based on batch assignments
        
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