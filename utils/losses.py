#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-objective loss functions for process mining
Combines task prediction, time estimation, and structural objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any


class ProcessLoss(nn.Module):
    """
    Multi-objective loss function for process mining
    Combines task prediction, time estimation, and structural objectives
    """
    def __init__(self, task_weight=0.5, time_weight=0.3, structure_weight=0.2,
                 class_weights=None, time_scale=1.0, balance_classes=True):
        """
        Initialize multi-objective loss
        
        Args:
            task_weight: Weight for task prediction loss
            time_weight: Weight for time prediction loss
            structure_weight: Weight for structural loss
            class_weights: Optional tensor of weights for task classes (for imbalanced data)
            time_scale: Scaling factor for time prediction to normalize magnitude
            balance_classes: Whether to use class-balanced loss for tasks
        """
        super().__init__()
        self.task_weight = task_weight
        self.time_weight = time_weight
        self.structure_weight = structure_weight
        self.time_scale = time_scale
        self.balance_classes = balance_classes
        
        # Task prediction loss - use CrossEntropyLoss with optional class weights
        self.task_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        
        # Time prediction loss - use Huber loss for robustness to outliers
        self.time_loss = nn.SmoothL1Loss(reduction='mean')
        
        # Keep track of running averages for normalization
        self.register_buffer('task_loss_avg', torch.tensor(1.0))
        self.register_buffer('time_loss_avg', torch.tensor(1.0))
        self.register_buffer('structure_loss_avg', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        
        # Momentum for running averages
        self.momentum = 0.9
    
    def forward(self, task_pred, task_target, time_pred=None, time_target=None, 
                embeddings=None, structure_info=None):
        """
        Compute multi-objective loss
        
        Args:
            task_pred: Task predictions [batch_size, num_classes]
            task_target: Task targets [batch_size]
            time_pred: Optional time predictions [batch_size]
            time_target: Optional time targets [batch_size]
            embeddings: Optional node embeddings [num_nodes, embedding_dim]
            structure_info: Optional structure information (dict with graph connectivity)
            
        Returns:
            Tuple of (combined_loss, loss_components_dict)
        """
        # Task prediction loss (always computed)
        task_loss = self.task_loss(task_pred, task_target)
        
        # Initialize other losses
        time_loss = torch.tensor(0.0, device=task_pred.device)
        structure_loss = torch.tensor(0.0, device=task_pred.device)
        
        # Time prediction loss (if provided)
        if time_pred is not None and time_target is not None:
            # Scale time values to avoid numerical issues
            time_pred_scaled = time_pred * self.time_scale
            time_target_scaled = time_target * self.time_scale
            time_loss = self.time_loss(time_pred_scaled, time_target_scaled)
        
        # Structural loss (if embeddings and structure info provided)
        if embeddings is not None and structure_info is not None:
            structure_loss = self._compute_structure_loss(embeddings, structure_info)
        
        # Update running averages for normalization (if in training mode)
        if self.training:
            self._update_averages(task_loss, time_loss, structure_loss)
            
            # Normalize losses by running averages
            task_loss_norm = task_loss / self.task_loss_avg.clamp(min=1e-8)
            time_loss_norm = time_loss / self.time_loss_avg.clamp(min=1e-8)
            structure_loss_norm = structure_loss / self.structure_loss_avg.clamp(min=1e-8)
        else:
            # During evaluation, use the stored averages
            task_loss_norm = task_loss / self.task_loss_avg.clamp(min=1e-8)
            time_loss_norm = time_loss / self.time_loss_avg.clamp(min=1e-8)
            structure_loss_norm = structure_loss / self.structure_loss_avg.clamp(min=1e-8)
        
        # Combine losses with weights
        combined_loss = (
            self.task_weight * task_loss_norm +
            self.time_weight * time_loss_norm +
            self.structure_weight * structure_loss_norm
        )
        
        # Return combined loss and components
        return combined_loss, {
            'task_loss': task_loss.item(),
            'time_loss': time_loss.item() if time_loss.numel() > 0 else 0.0,
            'structure_loss': structure_loss.item() if structure_loss.numel() > 0 else 0.0,
            'combined_loss': combined_loss.item()
        }
    
    def _compute_structure_loss(self, embeddings, structure_info):
        """
        Compute structural loss based on embeddings and graph structure
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            structure_info: Dictionary with graph structure information
            
        Returns:
            Structural loss tensor
        """
        loss = torch.tensor(0.0, device=embeddings.device)
        
        # Check for critical path information
        if 'critical_path' in structure_info:
            critical_path = structure_info['critical_path']
            if len(critical_path) > 1:
                # Get embeddings for critical path nodes
                cp_embeddings = embeddings[critical_path]
                
                # Minimize distance between consecutive nodes in critical path
                dist = torch.cdist(cp_embeddings.unsqueeze(0), cp_embeddings.unsqueeze(0)).squeeze(0)
                
                # Get distances between consecutive nodes in the path
                idx = torch.arange(len(critical_path) - 1, device=embeddings.device)
                consecutive_dist = dist[idx, idx + 1]
                
                # Critical path should have small distances between consecutive nodes
                critical_loss = consecutive_dist.mean()
                loss = loss + critical_loss
        
        # Check for node clusters information
        if 'node_clusters' in structure_info:
            node_clusters = structure_info['node_clusters']
            
            # Compute cluster-based loss if we have multiple clusters
            if len(node_clusters) > 1:
                cluster_loss = torch.tensor(0.0, device=embeddings.device)
                
                # Calculate cluster centroids
                centroids = []
                for cluster in node_clusters:
                    if cluster:  # Non-empty cluster
                        centroid = embeddings[cluster].mean(dim=0)
                        centroids.append(centroid)
                
                if centroids:
                    centroids = torch.stack(centroids)
                    
                    # Inter-cluster separation: maximize distance between centroids
                    centroid_dist = torch.pdist(centroids)
                    inter_cluster_loss = -centroid_dist.mean()  # Negative to maximize
                    
                    # Intra-cluster cohesion: minimize distance within clusters
                    intra_cluster_loss = torch.tensor(0.0, device=embeddings.device)
                    for i, cluster in enumerate(node_clusters):
                        if cluster and len(cluster) > 1:  # Need at least 2 nodes
                            cluster_embs = embeddings[cluster]
                            centroid = centroids[i]
                            
                            # Distance from nodes to their centroid
                            dists = torch.norm(cluster_embs - centroid.unsqueeze(0), dim=1)
                            intra_cluster_loss = intra_cluster_loss + dists.mean()
                    
                    # Balance inter and intra cluster objectives
                    cluster_loss = intra_cluster_loss + inter_cluster_loss
                    loss = loss + cluster_loss
        
        # Check for transition probabilities
        if 'transitions' in structure_info:
            transitions = structure_info['transitions']
            
            if transitions.numel() > 0:
                # Nodes with high transition probabilities should be close
                src, dst, prob = transitions[:,0], transitions[:,1], transitions[:,2]
                
                # Get embeddings
                src_emb = embeddings[src.long()]
                dst_emb = embeddings[dst.long()]
                
                # Compute distances weighted by transition probabilities
                distances = torch.norm(src_emb - dst_emb, dim=1)
                transition_loss = (distances * prob).mean()
                
                loss = loss + transition_loss
                
        return loss
    
    def _update_averages(self, task_loss, time_loss, structure_loss):
        """Update running averages of loss values for normalization"""
        self.update_count += 1
        
        # Task loss average
        self.task_loss_avg = self.momentum * self.task_loss_avg + \
                             (1 - self.momentum) * task_loss.detach()
        
        # Time loss average (if non-zero)
        if time_loss.numel() > 0 and time_loss > 0:
            self.time_loss_avg = self.momentum * self.time_loss_avg + \
                                (1 - self.momentum) * time_loss.detach()
        
        # Structure loss average (if non-zero)
        if structure_loss.numel() > 0 and structure_loss > 0:
            self.structure_loss_avg = self.momentum * self.structure_loss_avg + \
                                     (1 - self.momentum) * structure_loss.detach()


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification
    Focuses more on hard examples by down-weighting easy examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize focal loss
        
        Args:
            alpha: Optional class weights for imbalanced data [num_classes]
            gamma: Focusing parameter (higher means more focus on hard examples)
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Targets [batch_size]
            
        Returns:
            Loss value
        """
        # Get class probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get probability for the correct class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))
        target_probs = (probs * targets_one_hot).sum(1)
        
        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        # Compute loss
        loss = -focal_weight * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DualTaskLoss(nn.Module):
    """
    Loss function optimizing both next activity prediction and remaining time
    """
    def __init__(self, task_weight=0.7, time_weight=0.3, class_weights=None):
        """
        Initialize dual task loss
        
        Args:
            task_weight: Weight for task prediction loss
            time_weight: Weight for time prediction loss
            class_weights: Optional tensor of weights for task classes
        """
        super().__init__()
        self.task_weight = task_weight
        self.time_weight = time_weight
        
        # Task prediction loss with optional class weights
        if class_weights is not None:
            self.task_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.task_loss = nn.CrossEntropyLoss()
        
        # Time prediction loss with log transform for better scaling
        self.time_loss = LogCoshLoss()
    
    def forward(self, task_pred, task_target, time_pred, time_target):
        """
        Compute dual task loss
        
        Args:
            task_pred: Task predictions [batch_size, num_classes]
            task_target: Task targets [batch_size]
            time_pred: Time predictions [batch_size]
            time_target: Time targets [batch_size]
            
        Returns:
            Tuple of (combined_loss, loss_components_dict)
        """
        # Task prediction loss
        task_loss_val = self.task_loss(task_pred, task_target)
        
        # Time prediction loss (with log-transformation for better numerical properties)
        time_pred_pos = torch.clamp(time_pred, min=1e-8)  # Ensure positive values
        time_target_pos = torch.clamp(time_target, min=1e-8)  # Ensure positive values
        
        time_loss_val = self.time_loss(time_pred_pos, time_target_pos)
        
        # Combine losses
        combined_loss = self.task_weight * task_loss_val + self.time_weight * time_loss_val
        
        # Return combined loss and components
        return combined_loss, {
            'task_loss': task_loss_val.item(),
            'time_loss': time_loss_val.item(),
            'combined_loss': combined_loss.item()
        }


class LogCoshLoss(nn.Module):
    """
    Log-cosh loss function - smooth approximation of L1 loss with better gradients
    More robust to outliers than MSE
    """
    def __init__(self, reduction='mean'):
        """
        Initialize log-cosh loss
        
        Args:
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        """
        Compute log-cosh loss
        
        Args:
            y_pred: Predictions
            y_true: Targets
            
        Returns:
            Loss value
        """
        def _log_cosh(x):
            # Numerically stable implementation of log(cosh(x))
            # log(cosh(x)) = log((exp(x) + exp(-x))/2)
            #              = x + log(1 + exp(-2x)) - log(2)
            return x + F.softplus(-2.0 * x) - np.log(2.0)
        
        # Compute error
        diff = y_pred - y_true
        
        # Apply log-cosh
        loss = _log_cosh(diff)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class AdaptiveWeightedLoss(nn.Module):
    """
    Adaptive weighted loss that automatically adjusts component weights
    during training based on gradient magnitudes
    """
    def __init__(self, loss_modules, initial_weights=None, adaptive_factor=0.1):
        """
        Initialize adaptive weighted loss
        
        Args:
            loss_modules: Dictionary of loss modules {'name': loss_module}
            initial_weights: Initial weights for each loss component
            adaptive_factor: Factor controlling how quickly weights adapt
        """
        super().__init__()
        self.loss_modules = nn.ModuleDict(loss_modules)
        self.num_losses = len(loss_modules)
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = torch.ones(self.num_losses) / self.num_losses
        else:
            # Normalize weights to sum to 1
            initial_weights = torch.tensor(initial_weights) / sum(initial_weights)
        
        self.register_buffer('loss_weights', initial_weights)
        self.adaptive_factor = adaptive_factor
        
        # Keep track of moving averages of gradients
        self.register_buffer('grad_magnitudes', torch.ones(self.num_losses))
        self.register_buffer('update_count', torch.tensor(0))
        
        # Momentum for moving averages
        self.momentum = 0.9
    
    def forward(self, inputs_dict):
        """
        Compute adaptive weighted loss
        
        Args:
            inputs_dict: Dictionary of inputs for each loss component
                {loss_name: {'pred': predictions, 'target': targets}}
            
        Returns:
            Tuple of (combined_loss, loss_components_dict)
        """
        loss_values = {}
        combined_loss = 0.0
        
        # Compute all loss components
        for i, (name, loss_module) in enumerate(self.loss_modules.items()):
            if name in inputs_dict:
                # Extract inputs for this loss component
                inputs = inputs_dict[name]
                
                # Compute loss
                if isinstance(inputs, dict):
                    loss = loss_module(**inputs)
                else:
                    loss = loss_module(*inputs)
                
                # Store loss value
                loss_values[name] = loss.item()
                
                # Add weighted loss to combined loss
                combined_loss = combined_loss + self.loss_weights[i] * loss
            else:
                loss_values[name] = 0.0
        
        # Update weights in training mode
        if self.training and self.adaptive_factor > 0:
            self._update_weights(loss_values)
        
        # Return combined loss and components
        return combined_loss, loss_values
    
    def _update_weights(self, loss_values):
        """
        Update loss weights based on gradient magnitudes
        
        Args:
            loss_values: Dictionary of loss values
        """
        self.update_count += 1
        
        # Compute gradient magnitude for each loss
        grad_mags = []
        for i, (name, _) in enumerate(self.loss_modules.items()):
            if name in loss_values and loss_values[name] > 0:
                # Use loss value as proxy for gradient magnitude
                grad_mags.append(loss_values[name])
            else:
                grad_mags.append(self.grad_magnitudes[i].item())
        
        # Update moving average of gradient magnitudes
        new_magnitudes = torch.tensor(grad_mags, device=self.grad_magnitudes.device)
        self.grad_magnitudes = self.momentum * self.grad_magnitudes + \
                              (1 - self.momentum) * new_magnitudes
        
        # Compute inverse gradient magnitudes (smaller gradients need larger weights)
        inverse_mags = 1.0 / self.grad_magnitudes.clamp(min=1e-8)
        
        # Update weights based on inverse gradient magnitudes
        new_weights = inverse_mags / inverse_mags.sum()
        
        # Apply adaptive factor for smooth updates
        self.loss_weights = (1 - self.adaptive_factor) * self.loss_weights + \
                           self.adaptive_factor * new_weights


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    batch_size = 16
    num_classes = 10
    
    # Task prediction test
    task_pred = torch.randn(batch_size, num_classes).to(device)
    task_target = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Time prediction test
    time_pred = torch.rand(batch_size).to(device) * 10  # Random times between 0-10
    time_target = torch.rand(batch_size).to(device) * 10
    
    # Embeddings test (for structural loss)
    embeddings = torch.randn(50, 64).to(device)  # 50 nodes, 64-dim embeddings
    
    # Structure info for the loss
    structure_info = {
        'critical_path': [0, 1, 3, 5, 7],  # Example critical path
        'node_clusters': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],  # Example clusters
        'transitions': torch.tensor([[0, 1, 0.8], [1, 3, 0.6], [3, 5, 0.9]]).to(device)  # src, dst, prob
    }
    
    print("Testing ProcessLoss...")
    process_loss = ProcessLoss().to(device)
    
    # Test full loss computation
    loss, components = process_loss(
        task_pred, task_target, 
        time_pred, time_target,
        embeddings, structure_info
    )
    
    print(f"Combined loss: {loss.item()}")
    print(f"Loss components: {components}")
    
    print("\nTesting FocalLoss...")
    focal_loss = FocalLoss(gamma=2.0).to(device)
    focal_loss_val = focal_loss(task_pred, task_target)
    print(f"Focal loss: {focal_loss_val.item()}")
    
    print("\nTesting DualTaskLoss...")
    dual_loss = DualTaskLoss().to(device)
    dual_loss_val, dual_components = dual_loss(task_pred, task_target, time_pred, time_target)
    print(f"Dual task loss: {dual_loss_val.item()}")
    print(f"Dual loss components: {dual_components}")
    
    print("\nTesting AdaptiveWeightedLoss...")
    adaptive_loss = AdaptiveWeightedLoss({
        'task': nn.CrossEntropyLoss(),
        'time': nn.MSELoss()
    }).to(device)
    
    adaptive_inputs = {
        'task': (task_pred, task_target),
        'time': (time_pred, time_target)
    }
    
    adaptive_loss_val, adaptive_components = adaptive_loss(adaptive_inputs)
    print(f"Adaptive loss: {adaptive_loss_val.item()}")
    print(f"Adaptive components: {adaptive_components}")
    print(f"Adaptive weights: {adaptive_loss.loss_weights}")
    
    print("\nAll tests completed successfully!")