# src/processmine/utils/losses.py
"""
Multi-objective loss functions for process mining.
Combines task prediction, time estimation, and structural objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProcessLoss(nn.Module):
    """
    Multi-objective loss function for process mining.
    Combines task prediction, time estimation, and structural objectives.
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
    Focal loss for imbalanced classification.
    Focuses more on hard examples by down-weighting easy examples.
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