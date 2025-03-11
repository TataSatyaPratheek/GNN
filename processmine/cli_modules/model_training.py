#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training pipeline for ProcessMine CLI
"""

import os
import time
import torch
from termcolor import colored
import logging

from processmine.core.training import setup_optimizer_and_loss, evaluate_model
from processmine.data.preprocessing import compute_class_weights
from processmine.models.base import setup_phase1_model

# Try to import AMP training
try:
    from processmine.core.amp_training import train_with_amp
except ImportError:
    train_with_amp = None

# Try to import checkpoint manager
try:
    from processmine.utils.checkpoint import CheckpointManager, TrainingCheckpoint
except ImportError:
    CheckpointManager = None
    TrainingCheckpoint = None

logger = logging.getLogger(__name__)

def run_model_training(args, data, device, run_dir, models_dir=None, viz_dir=None):
    """
    Run model training and evaluation pipeline
    
    Args:
        args: Command line arguments
        data: Dictionary with processed data
        device: Computing device
        run_dir: Base results directory
        models_dir: Directory for saving models
        viz_dir: Directory for visualizations
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    start_time = time.time()
    
    df = data['df']
    graphs = data['graphs']
    task_encoder = data['task_encoder']
    resource_encoder = data['resource_encoder']
    
    # Compute class weights for imbalanced data
    print(colored("📊 Computing class weights for imbalanced data...", "cyan"))
    num_classes = len(task_encoder.classes_)
    class_weights = compute_class_weights(df, num_classes)
    
    # Setup model
    model = setup_phase1_model(args, df, task_encoder, resource_encoder, device)
    
    # Setup optimizer and loss function
    optimizer, criterion = setup_optimizer_and_loss(model, args, class_weights, device)
    
    # Setup checkpoint manager if available
    checkpoint_manager = None
    if CheckpointManager is not None and models_dir is not None:
        checkpoint_dir = os.path.join(models_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model_name=args.model_type,
            save_freq=5,  # Save every 5 epochs
            max_to_keep=3,
            resume=True
        )
    
    # Train with checkpoint support if available
    if checkpoint_manager is not None and TrainingCheckpoint is not None:
        with TrainingCheckpoint(model, optimizer, checkpoint_manager) as checkpoint:
            # If we resumed, start_epoch will be set
            start_epoch = checkpoint.epoch + 1
            
            # Train model
            if train_with_amp is not None and device.type == 'cuda':
                model, train_metrics = train_with_amp(
                    model=model,
                    train_loader=torch.utils.data.DataLoader(graphs[:int(len(graphs)*0.8)], batch_size=args.batch_size, shuffle=True),
                    val_loader=torch.utils.data.DataLoader(graphs[int(len(graphs)*0.8):], batch_size=args.batch_size, shuffle=False),
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    epochs=args.epochs,
                    patience=5
                )
            else:
                # Use standard training (from your existing code)
                from processmine.core.training import train_and_evaluate_model_phase1
                model, train_metrics = train_and_evaluate_model_phase1(
                    model, graphs, args, criterion, optimizer, device, run_dir, 
                    start_epoch=start_epoch
                )
            
            # Update final checkpoint
            checkpoint.update(args.epochs, train_metrics, is_best=True)
            
            # Export best model
            if models_dir:
                best_model_path = os.path.join(models_dir, f"best_{args.model_type}_model.pth")
                checkpoint_manager.export_best_model(best_model_path)
    else:
        # Standard training without checkpoints
        from processmine.core.training import train_and_evaluate_model_phase1
        model, train_metrics = train_and_evaluate_model_phase1(
            model, graphs, args, criterion, optimizer, device, run_dir
        )
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, graphs[int(len(graphs)*0.8):], criterion, device)
    
    # Combine metrics
    metrics = {**train_metrics, **test_metrics}
    metrics['training_time'] = time.time() - start_time
    
    return model, metrics