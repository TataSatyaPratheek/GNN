# src/processmine/utils/ablation.py
"""
Utilities for conducting systematic ablation studies.
Supports tracking and comparing different model configurations.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

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
        
        # Store ablation variants
        self.variants = []
        
        # Store evaluation function
        self.evaluation_fn = None
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    
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
        
        # Save base config
        base_config_path = os.path.join(self.output_dir, "configs", "base_config.json")
        with open(base_config_path, 'w') as f:
            json.dump(self.base_config, f, indent=2)
        
        # Evaluate base model
        logger.info("Evaluating base model")
        try:
            base_metrics = self.evaluation_fn(self.base_config)
            base_metrics['model_name'] = "base_model"
            base_metrics['model_type'] = self._get_model_type(self.base_config)
        except Exception as e:
            logger.error(f"Error evaluating base model: {e}")
            base_metrics = {'error': str(e), 'model_name': "base_model"}
        
        # Store all results
        all_results = [base_metrics]
        
        # Evaluate each variant
        for variant in self.variants:
            variant_name = variant['name']
            variant_changes = variant['changes']
            
            # Create variant config by applying changes to base config
            variant_config = self._apply_config_changes(self.base_config, variant_changes)
            
            # Save variant config
            variant_config_path = os.path.join(self.output_dir, "configs", f"{variant_name}_config.json")
            with open(variant_config_path, 'w') as f:
                json.dump(variant_config, f, indent=2)
            
            # Evaluate variant
            logger.info(f"Evaluating variant: {variant_name}")
            try:
                variant_metrics = self.evaluation_fn(variant_config)
                variant_metrics['model_name'] = variant_name
                variant_metrics['model_type'] = self._get_model_type(variant_config)
                all_results.append(variant_metrics)
            except Exception as e:
                logger.error(f"Error evaluating variant {variant_name}: {e}")
                all_results.append({'error': str(e), 'model_name': variant_name})
        
        # Create DataFrame with all results
        comparison_df = pd.DataFrame(all_results)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, "ablation_results.csv")
        comparison_df.to_csv(csv_path, index=False)
        
        # Generate comparison plot
        self.plot_ablation_results(comparison_df)
        
        # Generate report
        self.generate_report(comparison_df)
        
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
    
    def plot_ablation_results(self, comparison_df=None, metrics=None):
        """
        Plot ablation study results
        
        Args:
            comparison_df: DataFrame with comparison results (will reload from file if None)
            metrics: List of metrics to plot (all if None)
            
        Returns:
            Path to the saved plot
        """
        # Load data if not provided
        if comparison_df is None:
            csv_path = os.path.join(self.output_dir, "ablation_results.csv")
            if os.path.exists(csv_path):
                comparison_df = pd.read_csv(csv_path)
            else:
                logger.warning("Cannot plot ablation results: No data available")
                return None
        
        if comparison_df.empty:
            logger.warning("Cannot plot ablation results: No data available")
            return None
        
        # Determine metrics to plot
        if metrics is None:
            # Exclude non-metric columns
            exclude_cols = ['model_name', 'model_type', 'training_duration', 'timestamp', 'error']
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
    
    def generate_report(self, comparison_df=None):
        """
        Generate a comprehensive ablation study report
        
        Args:
            comparison_df: DataFrame with comparison results (will reload from file if None)
            
        Returns:
            Path to the saved report
        """
        # Load data if not provided
        if comparison_df is None:
            csv_path = os.path.join(self.output_dir, "ablation_results.csv")
            if os.path.exists(csv_path):
                comparison_df = pd.read_csv(csv_path)
            else:
                logger.warning("Cannot generate report: No data available")
                return None
        
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
            
            if not comparison_df.empty:
                f.write("### Performance Metrics\n\n")
                f.write(comparison_df.to_markdown(index=False))
                f.write("\n\n")
                
                # Calculate relative improvements
                if 'base_model' in comparison_df['model_name'].values:
                    f.write("### Relative Improvements\n\n")
                    base_row = comparison_df[comparison_df['model_name'] == 'base_model'].iloc[0]
                    
                    # Exclude non-metric columns
                    exclude_cols = ['model_name', 'model_type', 'training_duration', 'timestamp', 'error']
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