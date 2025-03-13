# models/base.py - Simplified model base
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional, Union

class ProcessModel(nn.Module):
    """Unified base class for all process mining models"""
    
    def __init__(self):
        super().__init__()
        self.is_neural = True  # Flag to distinguish from sklearn wrappers
    
    def get_param_count(self) -> int:
        """Return parameter count"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def predict(self, data):
        """
        Make predictions on input data
        Override in subclasses for specific prediction logic
        """
        raise NotImplementedError
    
    def get_metrics(self, true_labels, predictions) -> Dict[str, float]:
        """Calculate model performance metrics"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(true_labels, predictions, average='macro', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='macro', zero_division=0)
        }
        
        return metrics