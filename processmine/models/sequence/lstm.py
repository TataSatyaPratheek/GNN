#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM model for next activity prediction in process mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from processmine.models.base import SequenceModel

# Set up logging
logger = logging.getLogger(__name__)


class NextActivityLSTM(SequenceModel):
    """
    LSTM model for next activity prediction
    """
    def __init__(self, num_cls, emb_dim=64, hidden_dim=64, num_layers=1, dropout=0.3):
        """
        Initialize LSTM model
        
        Args:
            num_cls: Number of activity classes
            emb_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        self.num_cls = num_cls
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer for task IDs (add 1 for padding token)
        self.emb = nn.Embedding(num_cls+1, emb_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            emb_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Regularization
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_cls)

    def forward(self, x, seq_len=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len] with task IDs
                or PyG Data object
            seq_len: Sequence lengths [batch_size]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Handle PyG Data objects
        if hasattr(x, 'x') and hasattr(x, 'batch'):
            # Extract tasks and convert to sequences
            # This is just a placeholder, actual implementation would depend on
            # how sequence data is stored in the graph
            batch_size = torch.unique(x.batch).size(0)
            
            # Process node features in a sequence-like manner
            # This is a simplified approach and may not be optimal for all use cases
            return self._process_graph_data(x)
        
        # Sort sequences by length for more efficient packing
        if seq_len is not None:
            seq_len_sorted, perm_idx = seq_len.sort(0, descending=True)
            x_sorted = x[perm_idx]
            
            # Apply embedding
            x_emb = self.emb(x_sorted)
            
            # Pack sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                x_emb, seq_len_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # Process with LSTM
            out_packed, (h_n, c_n) = self.lstm(packed)
            
            # Get last hidden state
            last_hidden = h_n[-1]
            
            # Reorder to original order
            _, unperm_idx = perm_idx.sort(0)
            last_hidden = last_hidden[unperm_idx]
        else:
            # Standard processing without packing (for fixed-length sequences)
            x_emb = self.emb(x)
            _, (h_n, _) = self.lstm(x_emb)
            last_hidden = h_n[-1]
        
        # Apply batch normalization and dropout for more stable training
        last_hidden = self.batch_norm(last_hidden)
        last_hidden = self.dropout_layer(last_hidden)
        
        # Final prediction
        logits = self.fc(last_hidden)
        return logits
    
    def _process_graph_data(self, data):
        """
        Process PyG Data as sequence data
        
        Args:
            data: PyG Data object
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # This is a placeholder implementation and should be adapted for specific use cases
        # Extract features based on batch assignment
        x = data.x
        batch = data.batch
        
        # Get unique batches
        unique_batches = torch.unique(batch)
        batch_size = unique_batches.size(0)
        
        # Process each batch as a sequence
        hidden_states = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for i, b in enumerate(unique_batches):
            # Get nodes for this batch
            mask = (batch == b)
            batch_x = x[mask]
            
            # Embed features if needed
            if batch_x.dim() == 2 and batch_x.size(1) == 1:
                # Assume single task ID features
                batch_x = self.emb(batch_x.squeeze(1))
            
            # Use LSTM on this sequence
            _, (h_n, _) = self.lstm(batch_x.unsqueeze(0))  # Add batch dimension
            hidden_states[i] = h_n[-1]
        
        # Apply normalization and dropout
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)
        
        # Final prediction
        return self.fc(hidden_states)


def prepare_sequence_data(df, max_len=None, val_ratio=0.2):
    """
    Prepare sequence data for LSTM training with improved efficiency
    
    Args:
        df: Process data dataframe
        max_len: Maximum sequence length
        val_ratio: Validation set ratio
        
    Returns:
        Tuple of (train_sequences, test_sequences)
    """
    logger.info("Preparing sequence data")
    start_time = time.time()