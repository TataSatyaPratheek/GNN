#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for conducting systematic ablation studies
Supports tracking and comparing different model configurations
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            matthews_corrcoef, roc_auc_score, confusion_matrix)
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks metrics for ablation studies and model comparisons
    """
    def __init__(self, output_dir, experiment_name=None):
        """
        Initialize metrics tracker
        
        Args:
            output_dir: Directory to save results
            experiment_name: Optional experiment name
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        
        # Ensure output directory exists
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # Initialize metrics storage
        self.models = {}
        self.current_model = None
    
    def register_model(self, model_name, model_type, model_params=None, description=None):
        """
        Register a model for tracking
        
        Args:
            model_name: Name of the model (should be unique)
            model_type: Type of model (e.g., 'decision_tree', 'gnn', 'lstm')
            model_params: Optional dictionary of model parameters
            description: Optional model description
            
        Returns:
            model_name for method chaining
        """
        if model_name in self.models:
            logger.warning(f"Model '{model_name}' already registered. Overwriting.")
        
        self.models[model_name] = {
            'name': model_name,
            'type': model_type,
            'params': model_params or {},
            'description': description or '',
            'metrics': {},
            'training': {
                'start_time': None,
                'end_time': None,
                'duration': None,
                'epochs': [],
                'learning_curves': {}
            },
            'evaluation': {},
            'memory': {},
            'artifacts': {}
        }
        
        self.current_model = model_name
        logger.info(f"Registered model: {model_name} ({model_type})")
        
        return model_name
    
    def start_training(self, model_name=None):
        """
        Mark the start of model training
        
        Args:
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        self.models[model_name]['training']['start_time'] = time.time()
        logger.info(f"Started training for model: {model_name}")
        
        return self
    
    def end_training(self, model_name=None):
        """
        Mark the end of model training
        
        Args:
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        end_time = time.time()
        start_time = self.models[model_name]['training']['start_time']
        
        if start_time is None:
            logger.warning(f"Training start time not set for model '{model_name}'")
            duration = None
        else:
            duration = end_time - start_time
        
        self.models[model_name]['training']['end_time'] = end_time
        self.models[model_name]['training']['duration'] = duration
        
        logger.info(f"Ended training for model: {model_name}, duration: {duration:.2f}s")
        
        return self
    
    def log_epoch(self, epoch, metrics, model_name=None):
        """
        Log metrics for a training epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        epoch_data = {'epoch': epoch, **metrics}
        self.models[model_name]['training']['epochs'].append(epoch_data)
        
        # Update learning curves
        for metric_name, value in metrics.items():
            if metric_name not in self.models[model_name]['training']['learning_curves']:
                self.models[model_name]['training']['learning_curves'][metric_name] = []
            
            self.models[model_name]['training']['learning_curves'][metric_name].append(value)
        
        logger.debug(f"Logged epoch {epoch} for model: {model_name}")
        
        return self
    
    def log_evaluation(self, metrics, dataset_name='test', model_name=None):
        """
        Log evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            dataset_name: Name of the evaluation dataset
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        if dataset_name not in self.models[model_name]['evaluation']:
            self.models[model_name]['evaluation'][dataset_name] = {}
        
        # Add timestamp
        metrics['timestamp'] = time.time()
        
        # Update metrics
        self.models[model_name]['evaluation'][dataset_name].update(metrics)
        
        logger.info(f"Logged evaluation metrics for model: {model_name}, dataset: {dataset_name}")
        
        return self
    
    def log_memory_usage(self, memory_metrics, phase='inference', model_name=None):
        """
        Log memory usage metrics
        
        Args:
            memory_metrics: Dictionary of memory metrics
            phase: Phase of operation ('training', 'inference', etc.)
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        if phase not in self.models[model_name]['memory']:
            self.models[model_name]['memory'][phase] = {}
        
        # Add timestamp
        memory_metrics['timestamp'] = time.time()
        
        # Update metrics
        self.models[model_name]['memory'][phase].update(memory_metrics)
        
        logger.debug(f"Logged memory usage for model: {model_name}, phase: {phase}")
        
        return self
    
    def add_artifact(self, artifact_name, artifact_path, model_name=None):
        """
        Add an artifact for a model
        
        Args:
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        self.models[model_name]['artifacts'][artifact_name] = artifact_path
        
        logger.debug(f"Added artifact '{artifact_name}' for model: {model_name}")
        
        return self
    
    def log_prediction_metrics(self, y_true, y_pred, dataset_name='test', model_name=None):
        """
        Log comprehensive prediction metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            dataset_name: Name of the dataset
            model_name: Optional model name (uses current model if None)
            
        Returns:
            self for method chaining
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        # Convert to numpy arrays if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
        
        # Class-specific metrics
        classes = np.unique(np.concatenate([y_true, y_pred]))
        class_metrics = {}
        
        for cls in classes:
            # Binary metrics for this class (one-vs-rest)
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            
            # Skip classes with no true positives
            if np.sum(y_true_bin) == 0:
                continue
            
            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            
            class_metrics[f'class_{cls}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(y_true_bin))
            }
        
        metrics['class_metrics'] = class_metrics
        
        # Confusion matrix (store counts only)
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Log metrics
        self.log_evaluation(metrics, dataset_name, model_name)
        
        # Create and save confusion matrix visualization
        self._plot_confusion_matrix(cm, model_name, dataset_name)
        
        return self
    
    def _plot_confusion_matrix(self, cm, model_name, dataset_name):
        """Create and save confusion matrix visualization"""
        # Create directory if needed
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Normalize the confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot absolute counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_title(f"Confusion Matrix (Absolute Counts)")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        
        # Plot normalized values
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax2)
        ax2.set_title(f"Confusion Matrix (Normalized)")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        
        # Add title
        plt.suptitle(f"Model: {model_name}, Dataset: {dataset_name}")
        plt.tight_layout()
        
        # Save
        cm_path = os.path.join(viz_dir, f"{model_name}_{dataset_name}_cm.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Register artifact
        self.add_artifact(f'confusion_matrix_{dataset_name}', cm_path, model_name)
    
    def plot_learning_curves(self, model_name=None, metrics=None):
        """
        Plot learning curves for a model
        
        Args:
            model_name: Optional model name (uses current model if None)
            metrics: Optional list of metrics to plot (all if None)
            
        Returns:
            Path to the saved plot
        """
        model_name = model_name or self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered")
        
        learning_curves = self.models[model_name]['training']['learning_curves']
        
        if not learning_curves:
            logger.warning(f"No learning curves data for model '{model_name}'")
            return None
        
        # Determine which metrics to plot
        if metrics is None:
            metrics = list(learning_curves.keys())
        else:
            # Only include available metrics
            metrics = [m for m in metrics if m in learning_curves]
        
        if not metrics:
            logger.warning(f"No matching metrics found for model '{model_name}'")
            return None
        
        # Create learning curves plot
        n_metrics = len(metrics)
        fig_height = 4 * ((n_metrics + 1) // 2)  # Adjust height based on number of metrics
        
        fig, axes = plt.subplots(((n_metrics + 1) // 2), 2, figsize=(12, fig_height))
        fig.suptitle(f"Learning Curves: {model_name}", fontsize=16)
        
        # Handle case with only one metric
        if n_metrics == 1:
            axes = np.array([axes])
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = learning_curves[metric]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, 'b-', marker='o', markersize=4)
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust for suptitle
        
        # Save the plot
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        plot_path = os.path.join(viz_dir, f"{model_name}_learning_curves.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Register artifact
        self.add_artifact('learning_curves', plot_path, model_name)
        
        return plot_path
    
    def compare_models(self, model_names=None, metrics=None, dataset_name='test'):
        """
        Compare multiple models
        
        Args:
            model_names: List of model names to compare (all if None)
            metrics: List of metrics to compare (all common metrics if None)
            dataset_name: Name of the dataset to use for comparison
            
        Returns:
            DataFrame with comparison results
        """
        # Use all models if not specified
        if model_names is None:
            model_names = list(self.models.keys())
        
        # Ensure all models exist
        for model_name in model_names:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not registered")
        
        # Collect evaluation results
        results = []
        available_metrics = set()
        
        for model_name in model_names:
            if dataset_name in self.models[model_name]['evaluation']:
                eval_data = self.models[model_name]['evaluation'][dataset_name]
                
                # Only include non-nested metrics
                model_metrics = {k: v for k, v in eval_data.items() 
                               if not isinstance(v, (dict, list)) and k != 'timestamp'}
                
                available_metrics.update(model_metrics.keys())
                
                # Calculate size and training time
                training_data = self.models[model_name]['training']
                
                # Collect model info
                model_info = {
                    'model_name': model_name,
                    'model_type': self.models[model_name]['type'],
                    'training_duration': training_data.get('duration', None),
                    **model_metrics
                }
                
                results.append(model_info)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Filter metrics if specified
        if metrics is not None:
            columns = ['model_name', 'model_type', 'training_duration'] + [
                m for m in metrics if m in df.columns
            ]
            df = df[columns]
        
        # Plot comparison
        self._plot_model_comparison(df, metrics, dataset_name)
        
        return df
    
    def _plot_model_comparison(self, comparison_df, metrics=None, dataset_name='test'):
        """
        Plot model comparison
        
        Args:
            comparison_df: DataFrame with comparison results
            metrics: List of metrics to plot (all if None)
            dataset_name: Name of the dataset
        """
        if comparison_df.empty:
            logger.warning("Cannot plot comparison: No data available")
            return
        
        # Determine metrics to plot
        if metrics is None:
            # Exclude non-metric columns
            exclude_cols = ['model_name', 'model_type', 'training_duration', 'timestamp']
            metrics = [col for col in comparison_df.columns if col not in exclude_cols]
        
        # Set model_name as index for easier plotting
        df = comparison_df.set_index('model_name')
        
        # Create comparison plots
        n_metrics = len(metrics)
        if n_metrics == 0:
            logger.warning("No metrics to plot for comparison")
            return
        
        # Create directory if needed
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a bar plot for each metric
        plt.figure(figsize=(14, n_metrics * 3))
        
        for i, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
                
            plt.subplot(n_metrics, 1, i+1)
            ax = df[metric].sort_values(ascending=False).plot(kind='bar')
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.ylabel(metric)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on top of bars
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.4f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        comparison_path = os.path.join(viz_dir, f"model_comparison_{dataset_name}.png")
        plt.savefig(comparison_path)
        plt.close()
        
        # Create a summary CSV
        csv_path = os.path.join(self.output_dir, "metrics", f"model_comparison_{dataset_name}.csv")
        comparison_df.to_csv(csv_path, index=False)
    
    def save(self):
        """
        Save all tracking data to files
        
        Returns:
            Dictionary with paths to saved files
        """
        # Create directories if needed
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save individual model data
        for model_name, model_data in self.models.items():
            model_file = os.path.join(metrics_dir, f"{model_name}_metrics.json")
            
            # Convert any non-serializable values to strings
            json_data = self._make_json_serializable(model_data)
            
            with open(model_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            saved_files[model_name] = model_file
            logger.info(f"Saved metrics for model '{model_name}' to {model_file}")
        
        # Save summary
        summary_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'models': {
                model_name: {
                    'type': model_data['type'],
                    'description': model_data['description'],
                    'training_duration': model_data['training'].get('duration', None),
                    'evaluation': model_data['evaluation']
                }
                for model_name, model_data in self.models.items()
            }
        }
        
        summary_file = os.path.join(metrics_dir, "experiment_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        saved_files['summary'] = summary_file
        logger.info(f"Saved experiment summary to {summary_file}")
        
        # Create and save model comparison
        try:
            comparison_df = self.compare_models()
            if not comparison_df.empty:
                comparison_file = os.path.join(metrics_dir, "model_comparison.csv")
                comparison_df.to_csv(comparison_file, index=False)
                saved_files['comparison'] = comparison_file
                logger.info(f"Saved model comparison to {comparison_file}")
        except Exception as e:
            logger.warning(f"Could not create model comparison: {e}")
        
        return saved_files
    
    def _make_json_serializable(self, obj):
        """Convert a nested object to a JSON serializable format"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return str(obj)


class AblationManager:
    """
    Manages ablation studies by systematically testing model variants
    """
    def __init__(self, base_config, output_dir, experiment_name=None):
        """
        Initialize ablation study manager
        
        Args:
            base_config: Base configuration for the model
            output_dir: Directory to save results
            experiment_name: Optional experiment name
        """
        self.base_config = base_config
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"ablation_{int(time.time())}"
        
        # Initialize metrics tracker
        self.tracker = MetricsTracker(output_dir, experiment_name)
        
        # Store ablation variants
        self.variants = []
        
        # Store evaluation function
        self.evaluation_fn = None
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)
    
    def add_variant(self, name, config_changes, description=None):
        """
        Add a model variant for ablation study
        
        Args:
            name: Variant name
            config_changes: Dictionary of configuration changes to apply
            description: Optional variant description
            
        Returns:
            self for method chaining
        """
        self.variants.append({
            'name': name,
            'changes': config_changes,
            'description': description or f"Variant with modified {list(config_changes.keys())}"
        })
        
        logger.info(f"Added ablation variant: {name}")
        
        return self
    
    def set_evaluation_function(self, fn):
        """
        Set the evaluation function for the ablation study
        
        Args:
            fn: Evaluation function that takes a model config and returns metrics
            
        Returns:
            self for method chaining
        """
        self.evaluation_fn = fn
        return self
    
    def run(self):
        """
        Run the ablation study
        
        Returns:
            DataFrame with ablation study results
        """
        if self.evaluation_fn is None:
            raise ValueError("Evaluation function not set. Use set_evaluation_function() first.")
        
        # Register base model
        base_model_name = "base_model"
        self.tracker.register_model(
            base_model_name,
            self._get_model_type(self.base_config),
            self.base_config,
            "Base model configuration"
        )
        
        # Save base config
        base_config_path = os.path.join(self.output_dir, "configs", "base_config.json")
        with open(base_config_path, 'w') as f:
            json.dump(self.base_config, f, indent=2)
        
        # Evaluate base model
        logger.info("Evaluating base model")
        try:
            base_metrics = self.evaluation_fn(self.base_config)
            self.tracker.log_evaluation(base_metrics, 'ablation', base_model_name)
        except Exception as e:
            logger.error(f"Error evaluating base model: {e}")
            base_metrics = {'error': str(e)}
        
        # Evaluate each variant
        for variant in self.variants:
            variant_name = variant['name']
            variant_changes = variant['changes']
            
            # Create variant config by applying changes to base config
            variant_config = self._apply_config_changes(self.base_config, variant_changes)
            
            # Register variant
            self.tracker.register_model(
                variant_name,
                self._get_model_type(variant_config),
                variant_config,
                variant['description']
            )
            
            # Save variant config
            variant_config_path = os.path.join(self.output_dir, "configs", f"{variant_name}_config.json")
            with open(variant_config_path, 'w') as f:
                json.dump(variant_config, f, indent=2)
            
            # Evaluate variant
            logger.info(f"Evaluating variant: {variant_name}")
            try:
                variant_metrics = self.evaluation_fn(variant_config)
                self.tracker.log_evaluation(variant_metrics, 'ablation', variant_name)
            except Exception as e:
                logger.error(f"Error evaluating variant {variant_name}: {e}")
                variant_metrics = {'error': str(e)}
        
        # Generate comparison
        comparison_df = self.tracker.compare_models(metrics=None, dataset_name='ablation')
        
        # Save all results
        self.tracker.save()
        
        return comparison_df
    
    def _apply_config_changes(self, base_config, changes):
        """Apply changes to base configuration"""
        # Create a deep copy of the base config
        import copy
        config = copy.deepcopy(base_config)
        
        # Apply changes
        for key_path, value in changes.items():
            # Handle nested keys with dot notation
            if '.' in key_path:
                keys = key_path.split('.')
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
            else:
                config[key_path] = value
        
        return config
    
    def _get_model_type(self, config):
        """Extract model type from config"""
        return config.get('model_type', 'unknown')
    
    def plot_ablation_results(self, metrics=None):
        """
        Plot ablation study results
        
        Args:
            metrics: List of metrics to plot (all if None)
            
        Returns:
            Path to the saved plot
        """
        # Get comparison data
        comparison_df = self.tracker.compare_models(metrics=metrics, dataset_name='ablation')
        
        if comparison_df.empty:
            logger.warning("Cannot plot ablation results: No data available")
            return None
        
        # Determine metrics to plot
        if metrics is None:
            # Exclude non-metric columns
            exclude_cols = ['model_name', 'model_type', 'training_duration', 'timestamp']
            metrics = [col for col in comparison_df.columns if col not in exclude_cols]
        
        # Create directory if needed
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create a more detailed ablation plot
        n_metrics = len(metrics)
        if n_metrics == 0:
            logger.warning("No metrics to plot for ablation study")
            return None
        
        # Set up the plot
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, n_metrics * 4))
        fig.suptitle(f"Ablation Study Results: {self.experiment_name}", fontsize=16)
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric not in comparison_df.columns:
                continue
                
            ax = axes[i]
            
            # Sort by metric value
            sorted_df = comparison_df.sort_values(metric, ascending=False)
            
            # Plot bars
            bars = ax.bar(sorted_df['model_name'], sorted_df[metric], color='skyblue')
            
            # Highlight base model
            base_idx = sorted_df['model_name'].tolist().index('base_model') if 'base_model' in sorted_df['model_name'].tolist() else -1
            if base_idx >= 0:
                bars[base_idx].set_color('orange')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(sorted_df[metric]),
                       f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)
            
            # Customize plot
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_ylabel(metric)
            ax.set_xlabel('')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust for suptitle
        
        # Save the plot
        plot_path = os.path.join(viz_dir, f"ablation_results.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_report(self):
        """
        Generate a comprehensive ablation study report
        
        Returns:
            Path to the saved report
        """
        # Create directory if needed
        report_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate markdown report
        report_path = os.path.join(report_dir, f"ablation_report.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Ablation Study Report: {self.experiment_name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview section
            f.write("## Overview\n\n")
            f.write(f"This report presents the results of an ablation study with "
                   f"{len(self.variants) + 1} model variants.\n\n")
            
            # Base model section
            f.write("## Base Model Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.base_config, indent=2))
            f.write("\n```\n\n")
            
            # Model variants section
            f.write("## Model Variants\n\n")
            
            for variant in self.variants:
                f.write(f"### {variant['name']}\n\n")
                f.write(f"{variant['description']}\n\n")
                f.write("Changes from base model:\n\n")
                f.write("```json\n")
                f.write(json.dumps(variant['changes'], indent=2))
                f.write("\n```\n\n")
            
            # Results section
            f.write("## Results\n\n")
            
            # Get comparison data
            comparison_df = self.tracker.compare_models(dataset_name='ablation')
            
            if not comparison_df.empty:
                f.write("### Performance Metrics\n\n")
                f.write(comparison_df.to_markdown(index=False))
                f.write("\n\n")
                
                # Calculate relative improvements
                if 'base_model' in comparison_df['model_name'].values:
                    f.write("### Relative Improvements\n\n")
                    base_row = comparison_df[comparison_df['model_name'] == 'base_model'].iloc[0]
                    
                    # Exclude non-metric columns
                    exclude_cols = ['model_name', 'model_type', 'training_duration', 'timestamp']
                    metrics = [col for col in comparison_df.columns if col not in exclude_cols]
                    
                    relative_df = comparison_df.copy()
                    
                    for metric in metrics:
                        if metric in base_row and base_row[metric] != 0:
                            relative_df[f'{metric}_rel'] = (relative_df[metric] - base_row[metric]) / base_row[metric] * 100
                    
                    # Select relative columns for display
                    rel_cols = ['model_name'] + [f'{m}_rel' for m in metrics if f'{m}_rel' in relative_df.columns]
                    
                    if len(rel_cols) > 1:  # Make sure we have relative metrics
                        rel_display = relative_df[rel_cols].copy()
                        
                        # Format as percentage with +/- sign
                        for col in rel_cols[1:]:
                            rel_display[col] = rel_display[col].apply(lambda x: f"{'+' if x >= 0 else ''}{x:.2f}%")
                        
                        f.write(rel_display.to_markdown(index=False))
                        f.write("\n\n")
            else:
                f.write("No comparison data available.\n\n")
            
            # Conclusion section
            f.write("## Conclusion\n\n")
            f.write("*Add your interpretation and conclusions here.*\n\n")
            
            # Visualizations section
            viz_dir = os.path.join(self.output_dir, "visualizations")
            ablation_plot = os.path.join(viz_dir, "ablation_results.png")
            
            if os.path.exists(ablation_plot):
                f.write("## Visualizations\n\n")
                f.write("### Ablation Results\n\n")
                f.write(f"![Ablation Results](../visualizations/ablation_results.png)\n\n")
        
        logger.info(f"Generated ablation report: {report_path}")
        
        return report_path


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Create test output directory
    test_output_dir = "test_ablation_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Create a metrics tracker for testing
    tracker = MetricsTracker(test_output_dir, "test_experiment")
    
    # Register a test model
    tracker.register_model(
        "model_1",
        "gnn",
        {"hidden_dim": 64, "num_layers": 2},
        "Test GNN model"
    )
    
    # Log training metrics
    tracker.start_training()
    
    for epoch in range(1, 6):
        train_loss = 1.0 - 0.15 * epoch + 0.02 * np.random.randn()
        val_loss = 1.1 - 0.13 * epoch + 0.03 * np.random.randn()
        
        tracker.log_epoch(epoch, {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": 0.5 + 0.08 * epoch + 0.02 * np.random.randn()
        })
    
    tracker.end_training()
    
    # Log evaluation metrics
    y_true = np.random.randint(0, 5, size=100)
    y_pred = y_true.copy()
    
    # Introduce some errors
    error_idx = np.random.choice(len(y_true), size=20, replace=False)
    y_pred[error_idx] = np.random.randint(0, 5, size=20)
    
    tracker.log_prediction_metrics(y_true, y_pred)
    
    # Register another model for comparison
    tracker.register_model(
        "model_2",
        "lstm",
        {"hidden_dim": 128, "num_layers": 1},
        "Test LSTM model"
    )
    
    # Log some metrics for the second model
    tracker.log_evaluation({
        "accuracy": 0.84,
        "f1_macro": 0.82,
        "precision_macro": 0.81,
        "recall_macro": 0.80,
        "mcc": 0.79
    })
    
    # Plot learning curves
    tracker.plot_learning_curves()
    
    # Compare models
    comparison = tracker.compare_models()
    print("Model comparison:")
    print(comparison)
    
    # Save all data
    saved_files = tracker.save()
    print(f"Saved metrics to: {saved_files}")
    
    # Test ablation manager
    base_config = {
        "model_type": "gnn",
        "hidden_dim": 64,
        "num_layers": 2,
        "heads": 4,
        "dropout": 0.5,
        "learning_rate": 0.001
    }
    
    ablation = AblationManager(base_config, test_output_dir, "test_ablation")
    
    # Add variants
    ablation.add_variant(
        "no_heads",
        {"heads": 1},
        "Single attention head"
    )
    
    ablation.add_variant(
        "more_layers",
        {"num_layers": 3},
        "Deeper model with 3 layers"
    )
    
    ablation.add_variant(
        "lower_dropout",
        {"dropout": 0.2},
        "Lower dropout rate"
    )
    
    # Define evaluation function (mock for testing)
    def mock_evaluation(config):
        # Simulate metrics based on config
        base_accuracy = 0.8
        
        # Adjust metrics based on configuration
        mods = {
            "hidden_dim": 0.001 * (config["hidden_dim"] - 64),
            "num_layers": 0.02 * (config["num_layers"] - 2),
            "heads": 0.01 * (config["heads"] - 1),
            "dropout": -0.05 * (config["dropout"] - 0.5)
        }
        
        accuracy = base_accuracy + sum(mods.values()) + 0.01 * np.random.randn()
        
        return {
            "accuracy": accuracy,
            "f1_macro": accuracy - 0.02 + 0.01 * np.random.randn(),
            "precision_macro": accuracy - 0.03 + 0.01 * np.random.randn(),
            "recall_macro": accuracy - 0.01 + 0.01 * np.random.randn()
        }
    
    # Set evaluation function
    ablation.set_evaluation_function(mock_evaluation)
    
    # Run ablation study
    results = ablation.run()
    print("Ablation results:")
    print(results)
    
    # Plot ablation results
    ablation.plot_ablation_results()
    
    # Generate report
    report_path = ablation.generate_report()
    print(f"Generated ablation report: {report_path}")
    
    print("All tests completed successfully!")