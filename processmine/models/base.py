#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model interfaces and abstract classes for process mining models
"""

import torch
import torch.nn as nn
import abc
from typing import Dict, List, Tuple, Optional, Union, Any


class ProcessMiningModel(nn.Module, abc.ABC):
    """
    Abstract base class for all process mining models
    """
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass defined by subclasses"""
        pass
    
    @property
    def name(self) -> str:
        """Return model name"""
        return self.__class__.__name__
    
    def get_parameter_count(self) -> int:
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def memory_size_mb(self) -> float:
        """Approximate memory size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class GraphModel(ProcessMiningModel):
    """
    Base class for graph-based process mining models
    """
    
    @abc.abstractmethod
    def forward(self, data):
        """
        Forward pass for graph models
        
        Args:
            data: PyG Data object with x, edge_index, and batch attributes
            
        Returns:
            Model outputs
        """
        pass


class SequenceModel(ProcessMiningModel):
    """
    Base class for sequence-based process mining models
    """
    
    @abc.abstractmethod
    def forward(self, x, seq_len=None):
        """
        Forward pass for sequence models
        
        Args:
            x: Input tensor [batch_size, seq_len, features] or PyG Data
            seq_len: Optional sequence lengths [batch_size]
            
        Returns:
            Model outputs
        """
        pass


class BaselineModel(ProcessMiningModel):
    """
    Base class for traditional machine learning baseline models
    """
    
    @abc.abstractmethod
    def fit(self, X, y):
        """
        Fit model to data
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            self
        """
        pass
    
    @abc.abstractmethod
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    def forward(self, x):
        """
        Compatibility layer for PyTorch integration
        
        Args:
            x: Input tensor or PyG Data
            
        Returns:
            Predictions
        """
        # Handle PyG Data objects
        if hasattr(x, 'x'):
            # Extract features from PyG Data
            features = x.x.cpu().numpy()
            return torch.tensor(self.predict(features))
        
        # Handle tensor inputs
        if isinstance(x, torch.Tensor):
            return torch.tensor(self.predict(x.cpu().numpy()))
        
        # Direct numpy input
        return torch.tensor(self.predict(x))