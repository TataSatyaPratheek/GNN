"""
Memory-efficient LSTM model for next activity prediction in process mining.
Supports packed sequences, attention, and efficient batch processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from processmine.models.base import BaseModel

class NextActivityLSTM(BaseModel):
    """
    LSTM model for next activity prediction with memory optimization and attention mechanism
    """
    def __init__(
        self, 
        num_cls: int, 
        emb_dim: int = 64, 
        hidden_dim: int = 64, 
        num_layers: int = 1, 
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = True,
        use_layer_norm: bool = True,
        mem_efficient: bool = True
    ):
        """
        Initialize LSTM model
        
        Args:
            num_cls: Number of activity classes
            emb_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            use_layer_norm: Whether to use layer normalization
            mem_efficient: Whether to use memory-efficient implementation
        """
        super().__init__()
        self.num_cls = num_cls
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.mem_efficient = mem_efficient
        
        # Direction factor (1 for unidirectional, 2 for bidirectional)
        self.dir_factor = 2 if bidirectional else 1
        
        # Embedding layer for task IDs (add 1 for padding token)
        self.emb = nn.Embedding(num_cls+1, emb_dim, padding_idx=0)
        
        # LSTM layer with optimized hyperparameters
        self.lstm = nn.LSTM(
            emb_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Layer normalization (helps with convergence and gradient flow)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim * self.dir_factor)
        
        # Attention mechanism for focusing on relevant parts of sequence
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.dir_factor, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        
        # Regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer with smart initialization
        self.fc = nn.Linear(hidden_dim * self.dir_factor, num_cls)
        # Xavier initialization for better gradient flow
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, x, seq_lengths=None):
        """
        Forward pass with efficient sequence handling
        
        Args:
            x: Input tensor [batch_size, seq_len] with task IDs
            seq_lengths: Sequence lengths [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Handle PyG Data objects
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            # Extract case-level sequences using batch information
            return self._process_graph_data(x)
        
        # Handle list of variable-length sequences
        if isinstance(x, list):
            # This expects a list of tensors with variable lengths
            # First, we need to pad the sequences to the same length
            return self._process_sequence_list(x, seq_lengths)
        
        # Apply embedding to get [batch_size, seq_len, emb_dim]
        x_emb = self.emb(x)
        
        if seq_lengths is not None:
            # Use packed sequences for variable-length input
            # Sort sequences by length for more efficient packing
            seq_lengths_cpu = seq_lengths.cpu() if isinstance(seq_lengths, torch.Tensor) else seq_lengths
            sorted_len, indices = torch.LongTensor(seq_lengths_cpu).sort(0, descending=True)
            
            # Reorder input based on length
            x_emb = x_emb[indices]
            
            # Create packed sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                x_emb, sorted_len, batch_first=True, enforce_sorted=True
            )
            
            # Run through LSTM
            packed_output, (h_n, c_n) = self.lstm(packed)
            
            # Unpack output
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Reorder outputs back to original order
            _, reverse_indices = indices.sort(0)
            
            if self.bidirectional:
                # For bidirectional, we concatenate the last hidden state from both directions
                h_n = h_n.view(self.num_layers, 2, -1, self.hidden_dim)
                h_n = h_n[-1].transpose(0,1).contiguous().view(-1, self.hidden_dim * 2)
                last_hidden = h_n[reverse_indices]
            else:
                # For unidirectional, we take the last hidden state
                last_hidden = h_n[-1][reverse_indices]
            
            # If using attention, apply it on the unpacked output
            if self.use_attention:
                # Reorder output to match original sequence order
                output = output[reverse_indices]
                
                # Apply attention to get weighted output
                attn_weights = self.attention(output)
                self.attention_weights = attn_weights
                
                # Create attention mask for variable length sequences
                mask = torch.arange(output.size(1))[None, :] < torch.LongTensor(seq_lengths_cpu)[reverse_indices][:, None]
                mask = mask.to(output.device)
                
                # Apply mask (set attention weights to large negative for padding)
                attn_weights = attn_weights.squeeze(-1).masked_fill(~mask, float('-inf'))
                attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
                
                # Weighted sum of output based on attention
                attended_output = (output * attn_weights).sum(1)
                
                # Use attended output as final representation
                last_hidden = attended_output
        else:
            # For fixed-length sequences, use simpler approach
            output, (h_n, _) = self.lstm(x_emb)
            
            if self.bidirectional:
                # Concatenate both directions
                last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                # Get last hidden state
                last_hidden = h_n[-1]
            
            # Apply attention if enabled
            if self.use_attention:
                attn_weights = self.attention(output).squeeze(-1)
                self.attention_weights = attn_weights
                
                # Softmax to get attention distribution
                attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
                
                # Weighted sum of output based on attention
                attended_output = (output * attn_weights).sum(1)
                
                # Use attended output as final representation
                last_hidden = attended_output
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            last_hidden = self.layer_norm(last_hidden)
        
        # Apply dropout for regularization
        last_hidden = self.dropout_layer(last_hidden)
        
        # Final prediction layer
        logits = self.fc(last_hidden)
        
        return {"task_pred": logits}
    
    def _process_sequence_list(self, sequences, seq_lengths=None):
        """
        Process a list of variable-length sequence tensors
        
        Args:
            sequences: List of sequence tensors [batch_size] with shapes [seq_len]
            seq_lengths: Optional list of sequence lengths
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Get sequence lengths if not provided
        if seq_lengths is None:
            seq_lengths = [len(seq) for seq in sequences]
        
        # Get device from first sequence
        device = sequences[0].device if sequences else torch.device('cpu')
        
        # Find max length for padding
        max_len = max(seq_lengths)
        
        # Create padded batch
        batch_size = len(sequences)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        # Fill in sequences
        for i, seq in enumerate(sequences):
            length = seq_lengths[i]
            padded[i, :length] = seq[:length]
        
        # Process padded batch
        return self.forward(padded, seq_lengths)
    
    # processmine/models/sequence/lstm.py (UPDATED METHOD)

    def _process_graph_data(self, data):
        """
        Process DGL graph data by extracting case-level sequences with proper feature handling
        
        Args:
            data: DGL Graph object with batch information
                
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Extract batch information using DGL's API
        if not hasattr(data, 'batch_num_nodes'):
            # Single graph case
            batch_num_nodes = [data.num_nodes()]
            num_graphs = 1
        else:
            # Batched graph case
            batch_num_nodes = data.batch_num_nodes()
            num_graphs = len(batch_num_nodes)
        
        # Extract sequence for each case
        sequences = []
        seq_lengths = []
        
        # Process each graph in the batch
        node_offset = 0
        for i in range(num_graphs):
            num_nodes = batch_num_nodes[i]
            
            # Get nodes for this graph
            node_indices = list(range(node_offset, node_offset + num_nodes))
            
            # Extract node features
            if 'feat' in data.ndata:
                case_nodes = data.ndata['feat'][node_indices]
            else:
                # Fallback to other node features if available
                feature_keys = list(data.ndata.keys())
                if feature_keys:
                    case_nodes = data.ndata[feature_keys[0]][node_indices]
                else:
                    # No features available, create empty tensor
                    case_nodes = torch.zeros(num_nodes, 1, device=data.device)
            
            # Check if node order information is available from edge attributes
            # In DGL, we can use edge weights for this if available
            ordered_indices = node_indices
            if 'time' in data.edata or 'weight' in data.edata:
                # Try to get ordering from edge attributes
                # This requires edges to have temporal information
                try:
                    # For simple cases, sort by incoming edge's time attribute
                    u, v = data.in_edges(ordered_indices)
                    if 'time' in data.edata:
                        edge_times = data.edata['time'][data.edge_ids(u, v)]
                        # Sort by edge time to get proper sequence
                        _, sorted_order = torch.sort(edge_times)
                        ordered_indices = [ordered_indices[i] for i in sorted_order]
                except Exception as e:
                    # Fall back to original order if sorting fails
                    pass
            
            # Get ordered node features
            ordered_features = case_nodes[ordered_indices - node_offset]
            
            # Extract task IDs using the appropriate feature
            if ordered_features.size(1) > 1:
                # Determine which feature to use as task_id based on naming or position
                task_feature_idx = 0  # Default to first feature
                
                # Try to find a better feature if feature names are available
                if hasattr(data, 'feature_names') and len(data.feature_names) > 0:
                    for i, name in enumerate(data.feature_names):
                        if 'task' in name.lower():
                            task_feature_idx = i
                            break
                
                task_ids = ordered_features[:, task_feature_idx].long()
            else:
                # Single feature node - use directly
                task_ids = ordered_features.squeeze(-1).long()
            
            # Handle possible negative or out-of-range IDs
            task_ids = torch.clamp(task_ids, min=0, max=self.num_cls-1)
            
            # Store sequence and length
            sequences.append(task_ids)
            seq_lengths.append(len(task_ids))
            
            # Update offset
            node_offset += num_nodes
        
        # Process extracted sequences
        return self._process_sequence_list(sequences, seq_lengths)

    def get_embeddings(self, x, seq_lengths=None):
        """
        Extract sequence embeddings
        
        Args:
            x: Input sequences
            seq_lengths: Sequence lengths
            
        Returns:
            Sequence embeddings
        """
        self.eval()
        with torch.no_grad():
            # Process input to get embeddings
            if hasattr(x, 'x') and hasattr(x, 'batch'):
                # Handle PyG data
                return self._extract_embeddings_graph(x)
            
            # Handle list of sequences
            if isinstance(x, list):
                return self._extract_embeddings_list(x, seq_lengths)
            
            # Handle padded tensor
            x_emb = self.emb(x)
            
            if seq_lengths is not None:
                # Use packed sequences
                seq_lengths_cpu = seq_lengths.cpu() if isinstance(seq_lengths, torch.Tensor) else seq_lengths
                sorted_len, indices = torch.LongTensor(seq_lengths_cpu).sort(0, descending=True)
                
                # Reorder input
                x_emb = x_emb[indices]
                
                # Pack and process
                packed = nn.utils.rnn.pack_padded_sequence(
                    x_emb, sorted_len, batch_first=True, enforce_sorted=True
                )
                
                packed_output, _ = self.lstm(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                
                # Reorder back
                _, reverse_indices = indices.sort(0)
                output = output[reverse_indices]
                
                return output
            else:
                # Fixed-length sequences
                output, _ = self.lstm(x_emb)
                return output
    
    def _extract_embeddings_list(self, sequences, seq_lengths=None):
        """Extract embeddings from a list of sequence tensors"""
        # Get sequence lengths if not provided
        if seq_lengths is None:
            seq_lengths = [len(seq) for seq in sequences]
        
        # Find max length for padding
        max_len = max(seq_lengths)
        batch_size = len(sequences)
        device = sequences[0].device if sequences else torch.device('cpu')
        
        # Create padded batch
        padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        # Fill in sequences
        for i, seq in enumerate(sequences):
            length = seq_lengths[i]
            padded[i, :length] = seq[:length]
        
        # Get embeddings
        return self.get_embeddings(padded, seq_lengths)
    
    def _extract_embeddings_graph(self, data):
        """Extract embeddings from PyG graph data"""
        # Extract features and batch assignments
        x = data.x
        batch = data.batch
        
        # Get unique batch indices (cases)
        unique_batches = torch.unique(batch)
        
        # Extract sequence for each case
        sequences = []
        seq_lengths = []
        
        for b in unique_batches:
            # Get nodes for this case
            mask = (batch == b)
            case_nodes = x[mask]
            
            # Extract task IDs
            if case_nodes.size(1) > 1:
                task_ids = case_nodes[:, 0].long()
            else:
                task_ids = case_nodes.long()
            
            # Store sequence and length
            sequences.append(task_ids)
            seq_lengths.append(len(task_ids))
        
        # Get embeddings for extracted sequences
        return self._extract_embeddings_list(sequences, seq_lengths)
    
    def get_attention_weights(self, x, seq_lengths=None):
        """
        Get attention weights for interpretability
        
        Args:
            x: Input sequences
            seq_lengths: Sequence lengths
            
        Returns:
            Attention weights
        """
        if not self.use_attention:
            return None
        
        # Run forward pass to populate attention weights
        _ = self.forward(x, seq_lengths)
        
        return self.attention_weights

class EnhancedProcessRNN(BaseModel):
    """
    Enhanced RNN architecture combining LSTM, GRU, and Transformer layers
    for complex sequential process mining tasks
    """
    def __init__(
        self,
        num_cls: int,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_gru: bool = False,
        use_transformer: bool = True,
        num_heads: int = 4,
        use_time_features: bool = True,
        time_encoding_dim: int = 8,
        mem_efficient: bool = True
    ):
        """
        Initialize enhanced RNN model
        
        Args:
            num_cls: Number of activity classes
            emb_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            use_gru: Whether to use GRU instead of LSTM
            use_transformer: Whether to add transformer layers on top of RNN
            num_heads: Number of attention heads in transformer
            use_time_features: Whether to use time features
            time_encoding_dim: Dimension for time encoding
            mem_efficient: Whether to use memory-efficient implementation
        """
        super().__init__()
        self.num_cls = num_cls
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_gru = use_gru
        self.use_transformer = use_transformer
        self.use_time_features = use_time_features
        self.mem_efficient = mem_efficient
        
        # Embedding layer for task IDs
        self.task_emb = nn.Embedding(num_cls+1, emb_dim, padding_idx=0)
        
        # Time feature encoding
        if use_time_features:
            self.time_encoder = nn.Linear(1, time_encoding_dim)
            rnn_input_dim = emb_dim + time_encoding_dim
        else:
            rnn_input_dim = emb_dim
        
        # Create RNN layer (LSTM or GRU)
        if use_gru:
            self.rnn = nn.GRU(
                rnn_input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        else:
            self.rnn = nn.LSTM(
                rnn_input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
        
        # Bidirectional output size
        rnn_output_dim = hidden_dim * 2
        
        # Optional transformer layers
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=rnn_output_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(rnn_output_dim)
        
        # Global attention for sequence-level representation
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.task_pred = nn.Linear(rnn_output_dim, num_cls)
        
        # Optional time prediction
        self.time_pred = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        # Xavier initialization for linear layers
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, batch):
        """
        Forward pass with support for different input formats
        
        Args:
            batch: Input batch which can be:
                - PyG Data object
                - Tuple of (task_ids, time_diffs)
                - Tensor of task IDs
                - Dictionary with 'task_ids' and optional 'time_diffs'
            
        Returns:
            Dictionary with predictions
        """
        # Extract inputs based on batch type
        if isinstance(batch, dict):
            # Dictionary input
            task_ids = batch['task_ids']
            time_diffs = batch.get('time_diffs') if self.use_time_features else None
            padding_mask = batch.get('padding_mask')
            seq_lengths = batch.get('seq_lengths')
        elif isinstance(batch, tuple) and len(batch) >= 2:
            # Tuple input (task_ids, time_diffs)
            task_ids = batch[0]
            time_diffs = batch[1] if self.use_time_features else None
            padding_mask = batch[2] if len(batch) > 2 else None
            seq_lengths = batch[3] if len(batch) > 3 else None
        elif hasattr(batch, 'x') and hasattr(batch, 'batch'):
            # PyG Data object
            return self._process_graph_data(batch)
        else:
            # Assume tensor input of task IDs
            task_ids = batch
            time_diffs = None
            padding_mask = None
            seq_lengths = None
        
        # Create embeddings
        task_embeddings = self.task_emb(task_ids)
        
        # Add time features if available
        if self.use_time_features and time_diffs is not None:
            # Ensure time_diffs has right shape [batch, seq, 1]
            if time_diffs.dim() == 2:
                time_diffs = time_diffs.unsqueeze(-1)
            
            # Encode time differences
            time_features = self.time_encoder(time_diffs)
            
            # Concatenate with task embeddings
            inputs = torch.cat([task_embeddings, time_features], dim=-1)
        else:
            inputs = task_embeddings
        
        # Handle variable-length sequences with packing
        if seq_lengths is not None:
            # Sort sequences by length
            seq_lengths_cpu = seq_lengths.cpu() if isinstance(seq_lengths, torch.Tensor) else seq_lengths
            sorted_len, indices = torch.LongTensor(seq_lengths_cpu).sort(0, descending=True)
            
            # Reorder input
            inputs = inputs[indices]
            
            # Pack sequences
            packed_inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, sorted_len, batch_first=True, enforce_sorted=True
            )
            
            # Process with RNN
            if isinstance(self.rnn, nn.LSTM):
                packed_outputs, (hidden, _) = self.rnn(packed_inputs)
            else:
                packed_outputs, hidden = self.rnn(packed_inputs)
            
            # Unpack outputs
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            
            # Reorder outputs to original order
            _, reverse_indices = indices.sort(0)
            outputs = outputs[reverse_indices]
            
            # Create padding mask
            if padding_mask is None:
                # Create mask from sequence lengths
                max_len = outputs.size(1)
                batch_size = outputs.size(0)
                padding_mask = torch.arange(max_len, device=outputs.device)[None, :] >= torch.tensor(
                    seq_lengths_cpu, device=outputs.device
                )[:, None]
        else:
            # Process fixed-length sequences
            if isinstance(self.rnn, nn.LSTM):
                outputs, (hidden, _) = self.rnn(inputs)
            else:
                outputs, hidden = self.rnn(inputs)
        
        # Apply transformer if enabled
        if self.use_transformer:
            if padding_mask is not None:
                # Apply transformer with mask
                transformer_outputs = self.transformer(outputs, src_key_padding_mask=padding_mask)
            else:
                transformer_outputs = self.transformer(outputs)
            
            # Residual connection
            outputs = outputs + transformer_outputs
        
        # Apply layer normalization
        outputs = self.layer_norm(outputs)
        
        # Apply attention to get sequence representation
        if padding_mask is not None:
            # Create attention scores
            attention_scores = self.attention(outputs).squeeze(-1)
            
            # Apply mask (set padding positions to large negative values)
            attention_scores = attention_scores.masked_fill(padding_mask, float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        else:
            attention_scores = self.attention(outputs).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        
        # Weighted sum to get sequence representation
        sequence_repr = (outputs * attention_weights).sum(1)
        
        # Apply dropout
        sequence_repr = self.dropout(sequence_repr)
        
        # Task prediction
        task_logits = self.task_pred(sequence_repr)
        
        # Time prediction if time features are used
        if self.use_time_features:
            time_pred = self.time_pred(sequence_repr).squeeze(-1)
            return {
                "task_pred": task_logits,
                "time_pred": time_pred,
                "attention_weights": attention_weights
            }
        else:
            return {
                "task_pred": task_logits,
                "attention_weights": attention_weights
            }
    
    def _process_graph_data(self, data):
        """
        Process DGL graph data by extracting case-level sequences with proper feature handling
        
        Args:
            data: DGL Graph object with batch information
                
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Get batch size
        batch_size = data.batch_size if hasattr(data, 'batch_size') else 1
        
        # Extract sequences for each graph in the batch
        sequences = []
        seq_lengths = []
        
        if batch_size > 1:
            # Get the number of nodes per graph
            batch_num_nodes = data.batch_num_nodes()
            
            # Track node offset for each graph
            node_offset = 0
            
            # Process each graph
            for i, num_nodes in enumerate(batch_num_nodes):
                # Get nodes for this graph
                graph_nodes = data.ndata['feat'][node_offset:node_offset + num_nodes]
                
                # Get task IDs - assume they're in first column or stored separately
                if graph_nodes.size(1) > 1:
                    task_ids = graph_nodes[:, 0].long()  # Use first feature as task ID
                else:
                    task_ids = graph_nodes.squeeze(-1).long()
                
                # Get sequence ordering - try to use position attribute if available
                if 'pos' in data.ndata:
                    pos = data.ndata['pos'][node_offset:node_offset + num_nodes]
                    # Sort by position
                    _, sorted_indices = torch.sort(pos.view(-1))
                    task_ids = task_ids[sorted_indices]
                
                # Handle possible out-of-range IDs
                task_ids = torch.clamp(task_ids, min=0, max=self.num_cls-1)
                
                # Store sequence and length
                sequences.append(task_ids)
                seq_lengths.append(len(task_ids))
                
                # Update offset
                node_offset += num_nodes
        else:
            # Single graph
            graph_nodes = data.ndata['feat']
            
            # Get task IDs
            if graph_nodes.size(1) > 1:
                task_ids = graph_nodes[:, 0].long()
            else:
                task_ids = graph_nodes.squeeze(-1).long()
            
            # Get sequence ordering if available
            if 'pos' in data.ndata:
                pos = data.ndata['pos']
                _, sorted_indices = torch.sort(pos.view(-1))
                task_ids = task_ids[sorted_indices]
            
            # Handle possible out-of-range IDs
            task_ids = torch.clamp(task_ids, min=0, max=self.num_cls-1)
            
            # Store sequence and length
            sequences.append(task_ids)
            seq_lengths.append(len(task_ids))
        
        # Process extracted sequences using existing function
        return self._process_sequence_list(sequences, seq_lengths)