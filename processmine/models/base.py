#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model classes for process mining with standardized interface
"""

import torch
import torch.nn as nn
import abc
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

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
    
    def predict(self, data_loader, device=None):
        """
        Make predictions using the model
        
        Args:
            data_loader: DataLoader containing the data
            device: Device to use for computation
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        self.eval()
        y_pred = []
        y_prob = []
        y_true = []
        
        # Use model's device if none specified
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            for batch_data in data_loader:
                # Prepare batch
                batch_data = self._prepare_batch(batch_data, device)
                
                # Get model outputs
                outputs = self(batch_data)
                
                # Extract predictions, probabilities, and labels
                batch_pred, batch_prob, batch_true = self._extract_predictions(outputs, batch_data)
                
                # Append to lists
                y_pred.extend(batch_pred)
                y_prob.extend(batch_prob)
                y_true.extend(batch_true)
        
        return y_pred, y_prob, y_true
    
    def fit(self, train_loader, val_loader=None, optimizer=None, criterion=None, 
            epochs=10, device=None, early_stopping=True, patience=5, 
            model_path=None, scheduler=None, callbacks=None):
        """
        Train the model with standardized process and early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            criterion: Loss function
            epochs: Number of epochs to train
            device: Device to use for training
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            model_path: Path to save best model
            scheduler: Learning rate scheduler
            callbacks: List of callback functions to call after each epoch
            
        Returns:
            self (for method chaining)
        """
        from tqdm import tqdm
        import time
        
        # Use model's device if none specified
        if device is None:
            device = next(self.parameters()).device
        
        # Create default optimizer if none provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Create default criterion if none provided
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        self.loss_history = []
        self.val_loss_history = []
        
        # For early stopping
        best_model_state = None
        
        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            self.train()
            train_loss = 0.0
            train_batch_count = 0
            
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
            
            for batch_data in train_progress:
                # Move batch to device
                batch_data = self._prepare_batch(batch_data, device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute loss
                loss = self._compute_loss(batch_data, criterion)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Track loss
                train_loss += loss.item()
                train_batch_count += 1
                
                # Update progress bar
                train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Calculate average training loss
            avg_train_loss = train_loss / max(1, train_batch_count)
            self.loss_history.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, metrics = self.evaluate(val_loader, criterion, device)
                self.val_loss_history.append(val_loss)
                
                # Update LR scheduler if using 'plateau'
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
            else:
                val_loss = avg_train_loss
                metrics = {}
            
            # Update LR scheduler if not using 'plateau'
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"time={epoch_time:.2f}s"
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model state for later use
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                
                # Save best model to disk if path provided
                if model_path:
                    torch.save(self.state_dict(), model_path)
                    logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter}/{patience} epochs")
                
                if early_stopping and patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Run callbacks if provided
            if callbacks:
                for callback in callbacks:
                    callback(
                        epoch=epoch, 
                        model=self, 
                        train_loss=avg_train_loss, 
                        val_loss=val_loss,
                        metrics=metrics, 
                        optimizer=optimizer
                    )
        
        # Load best model state if available and using early stopping
        if early_stopping and best_model_state:
            self.load_state_dict(best_model_state)
            logger.info("Restored best model state")
        
        return self
    
    def evaluate(self, data_loader, criterion=None, device=None):
        """
        Evaluate the model on a dataset
        
        Args:
            data_loader: DataLoader containing the data
            criterion: Loss function
            device: Device to use for computation
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.eval()
        
        # Use model's device if none specified
        if device is None:
            device = next(self.parameters()).device
        
        # Default criterion if none provided
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        
        total_loss = 0.0
        batch_count = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                # Prepare batch
                batch_data = self._prepare_batch(batch_data, device)
                
                # Compute loss
                loss = self._compute_loss(batch_data, criterion)
                
                # Get predictions
                outputs = self(batch_data)
                batch_preds, _, batch_targets = self._extract_predictions(outputs, batch_data)
                
                # Track metrics
                total_loss += loss.item()
                batch_count += 1
                
                # Collect predictions and targets
                all_preds.extend(batch_preds)
                all_targets.extend(batch_targets)
        
        # Calculate average loss
        avg_loss = total_loss / max(1, batch_count)
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions, targets):
        """
        Calculate metrics based on predictions and targets
        
        Args:
            predictions: List of predictions
            targets: List of true values
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        metrics = {}
        
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            # Convert to numpy arrays if needed
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(targets, predictions)
            
            # Multi-class metrics with error handling
            try:
                metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
                metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
                metrics['precision'] = precision_score(targets, predictions, average='macro', zero_division=0)
                metrics['recall'] = recall_score(targets, predictions, average='macro', zero_division=0)
            except Exception as e:
                logger.warning(f"Error calculating F1/precision/recall: {e}")
        
        except ImportError:
            # Fall back to basic accuracy if sklearn not available
            correct = sum(p == t for p, t in zip(predictions, targets))
            total = len(targets)
            metrics['accuracy'] = correct / total if total > 0 else 0
        
        return metrics
    
    @abc.abstractmethod
    def _prepare_batch(self, batch_data, device):
        """
        Prepare batch for model processing (move to device, etc.)
        
        Args:
            batch_data: Batch data
            device: Device to move data to
            
        Returns:
            Prepared batch data
        """
        pass
    
    @abc.abstractmethod
    def _compute_loss(self, batch_data, criterion):
        """
        Compute loss for a batch
        
        Args:
            batch_data: Batch data
            criterion: Loss function
            
        Returns:
            Loss tensor
        """
        pass
    
    @abc.abstractmethod
    def _extract_predictions(self, outputs, batch_data):
        """
        Extract predictions, probabilities, and true labels
        
        Args:
            outputs: Model outputs
            batch_data: Batch data
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        pass


class GraphModel(ProcessMiningModel):
    """
    Base class for graph-based process mining models
    """
    
    def _prepare_batch(self, batch_data, device):
        """Prepare PyG batch data for model"""
        return batch_data.to(device)
    
    def _compute_loss(self, batch_data, criterion):
        """Compute loss for PyG batch with graph-level targets"""
        # Get model outputs
        outputs = self(batch_data)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # Enhanced model with multiple outputs
            if "task_pred" in outputs:
                # Extract graph-level targets
                from torch_geometric.utils import scatter
                
                graph_targets = scatter(
                    batch_data.y,
                    batch_data.batch,
                    dim=0,
                    reduce="mode"
                )
                
                # Basic loss calculation
                loss = criterion(outputs["task_pred"], graph_targets)
                
                # Add diversity loss if available
                if "diversity_loss" in outputs:
                    loss = loss + outputs["diversity_loss"]
                
                return loss
        
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            # Tuple output format (logits, aux_loss)
            logits, aux_loss = outputs
            
            # Extract graph-level targets
            from torch_geometric.utils import scatter
            
            graph_targets = scatter(
                batch_data.y,
                batch_data.batch,
                dim=0,
                reduce="mode"
            )
            
            # Compute main loss and add auxiliary loss
            loss = criterion(logits, graph_targets) + aux_loss
            return loss
        
        # Default case - standard output format
        from torch_geometric.utils import scatter
        
        graph_targets = scatter(
            batch_data.y,
            batch_data.batch,
            dim=0,
            reduce="mode"
        )
        
        return criterion(outputs, graph_targets)
    
    def _extract_predictions(self, outputs, batch_data):
        """Extract predictions from PyG batch outputs"""
        import torch.nn.functional as F
        from torch_geometric.utils import scatter
        
        # Extract graph-level targets
        graph_targets = scatter(
            batch_data.y,
            batch_data.batch,
            dim=0,
            reduce="mode"
        ).cpu().numpy()
        
        # Handle different output formats
        if isinstance(outputs, dict):
            if "task_pred" in outputs:
                logits = outputs["task_pred"]
            else:
                logits = outputs.get("logits", outputs.get("output", None))
                
                if logits is None:
                    raise ValueError("Could not find predictions in model outputs")
        
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            # (logits, aux_loss) format
            logits = outputs[0]
        
        else:
            # Standard output
            logits = outputs
        
        # Get class probabilities
        probs = F.softmax(logits, dim=1).cpu().numpy()
        
        # Get predicted classes
        preds = logits.argmax(dim=1).cpu().numpy()
        
        return preds, probs, graph_targets


class SequenceModel(ProcessMiningModel):
    """
    Base class for sequence-based process mining models
    """
    
    def _prepare_batch(self, batch_data, device):
        """Prepare sequence batch data for model"""
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            # Tuple of (x, y)
            x, y = batch_data
            return (x.to(device), y.to(device))
        elif isinstance(batch_data, tuple) and len(batch_data) == 3:
            # Tuple of (x, lengths, y)
            x, lengths, y = batch_data
            return (x.to(device), lengths, y.to(device))
        else:
            # PyG data or other format
            return batch_data.to(device)
    
    def _compute_loss(self, batch_data, criterion):
        """Compute loss for sequence batch"""
        # Handle different batch formats
        if isinstance(batch_data, tuple):
            if len(batch_data) == 2:
                # (x, y) format
                x, y = batch_data
                outputs = self(x)
                return criterion(outputs, y)
            elif len(batch_data) == 3:
                # (x, lengths, y) format
                x, lengths, y = batch_data
                outputs = self(x, lengths)
                return criterion(outputs, y)
        
        # Default PyG data format
        outputs = self(batch_data)
        return criterion(outputs, batch_data.y)
    
    def _extract_predictions(self, outputs, batch_data):
        """Extract predictions from sequence model outputs"""
        import torch.nn.functional as F
        
        # Get true labels
        if isinstance(batch_data, tuple):
            if len(batch_data) == 2:
                # (x, y) format
                _, y = batch_data
            elif len(batch_data) == 3:
                # (x, lengths, y) format
                _, _, y = batch_data
            else:
                y = batch_data[-1]  # Assume last element is labels
            
            true_labels = y.cpu().numpy()
        else:
            # PyG data or other format
            true_labels = batch_data.y.cpu().numpy()
        
        # Get probabilities
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        # Get predictions
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        return preds, probs, true_labels


class TabularModel(ProcessMiningModel):
    """
    Base class for tabular data models (MLP, Scikit-learn wrappers)
    """
    
    def _prepare_batch(self, batch_data, device):
        """Prepare tabular batch data for model"""
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            # Tuple of (x, y)
            x, y = batch_data
            return (x.to(device), y.to(device))
        else:
            # PyG data or other format
            return batch_data.to(device)
    
    def _compute_loss(self, batch_data, criterion):
        """Compute loss for tabular batch"""
        # Handle different batch formats
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            # (x, y) format
            x, y = batch_data
            outputs = self(x)
            return criterion(outputs, y)
        
        # Default PyG data format
        outputs = self(batch_data.x)
        return criterion(outputs, batch_data.y)
    
    def _extract_predictions(self, outputs, batch_data):
        """Extract predictions from tabular model outputs"""
        import torch.nn.functional as F
        
        # Get true labels
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            # (x, y) format
            _, y = batch_data
            true_labels = y.cpu().numpy()
        else:
            # PyG data or other format
            true_labels = batch_data.y.cpu().numpy()
        
        # Get probabilities
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        # Get predictions
        preds = outputs.argmax(dim=1).cpu().numpy()
        
        return preds, probs, true_labels


class SklearnModel(ProcessMiningModel):
    """
    Wrapper for scikit-learn models to use with the same interfaces
    """
    
    def __init__(self, model):
        """
        Initialize sklearn model wrapper
        
        Args:
            model: Scikit-learn model instance
        """
        super().__init__()
        self.model = model
        self._is_fitted = False
    
    def forward(self, x):
        """Forward pass for compatibility with PyTorch"""
        # Convert to numpy if tensor
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Predict probabilities if available
        if hasattr(self.model, 'predict_proba') and self._is_fitted:
            proba = self.model.predict_proba(x_np)
            return torch.tensor(proba, dtype=torch.float32)
        
        # Predict classes if probabilities not available
        if self._is_fitted:
            preds = self.model.predict(x_np)
            # Convert to one-hot
            n_classes = len(np.unique(preds))
            one_hot = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                one_hot[i, p] = 1
            return torch.tensor(one_hot, dtype=torch.float32)
        
        # Not fitted yet
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)
    
    def fit(self, train_loader, val_loader=None, **kwargs):
        """
        Fit sklearn model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (ignored)
            **kwargs: Additional arguments passed to sklearn's fit method
            
        Returns:
            self for method chaining
        """
        # Extract data from loader
        X_train, y_train = self._extract_data_from_loader(train_loader)
        
        # Fit model
        self.model.fit(X_train, y_train, **kwargs)
        self._is_fitted = True
        
        return self
    
    def predict(self, data_loader, device=None):
        """
        Make predictions using the sklearn model
        
        Args:
            data_loader: DataLoader containing the data
            device: Device (ignored for sklearn models)
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        # Extract data from loader
        X, y_true = self._extract_data_from_loader(data_loader)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X)
        else:
            # Create simple probabilities (1 for predicted class)
            n_classes = len(np.unique(y_pred))
            y_prob = np.zeros((len(y_pred), n_classes))
            for i, p in enumerate(y_pred):
                y_prob[i, p] = 1
        
        return y_pred, y_prob, y_true
    
    def evaluate(self, data_loader, criterion=None, device=None):
        """
        Evaluate sklearn model
        
        Args:
            data_loader: DataLoader containing the data
            criterion: Loss function (ignored for sklearn models)
            device: Device (ignored for sklearn models)
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        # Make predictions
        y_pred, _, y_true = self.predict(data_loader)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_pred, y_true)
        
        # Approximate loss
        loss = 1.0 - metrics.get('accuracy', 0.0)
        
        return loss, metrics
    
    def _extract_data_from_loader(self, data_loader):
        """
        Extract features and labels from a data loader
        
        Args:
            data_loader: DataLoader
            
        Returns:
            Tuple of (features, labels)
        """
        X, y = [], []
        
        for batch in data_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                # (x, y) format
                batch_x, batch_y = batch
                X.extend(batch_x.cpu().numpy())
                y.extend(batch_y.cpu().numpy())
            else:
                # PyG data
                X.extend(batch.x.cpu().numpy())
                y.extend(batch.y.cpu().numpy())
        
        return np.array(X), np.array(y)
    
    def _prepare_batch(self, batch_data, device):
        """Prepare batch (no-op for sklearn)"""
        return batch_data
    
    def _compute_loss(self, batch_data, criterion):
        """Compute loss (dummy implementation for compatibility)"""
        return torch.tensor(0.0)
    
    def _extract_predictions(self, outputs, batch_data):
        """Extract predictions (dummy implementation for compatibility)"""
        # This will usually not be called directly
        return [], [], []