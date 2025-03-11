#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment management for process mining
Handles result directories, metrics tracking, and experiment lifecycle
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
import numpy as np
from termcolor import colored

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def setup_results_dir(custom_dir=None):
    """
    Create organized results directory structure with timestamp
    
    Args:
        custom_dir: Optional custom directory path
    
    Returns:
        Path to results directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use absolute path with optional custom directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "results")
    
    if custom_dir:
        if os.path.isabs(custom_dir):
            run_dir = custom_dir
        else:
            run_dir = os.path.join(script_dir, "..", "..", custom_dir)
    else:
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create subdirectories with descriptive names
    subdirs = {
        "models": "Saved model weights and parameters",
        "visualizations": "Generated plots and diagrams",
        "metrics": "Performance metrics and statistics",
        "analysis": "Process mining analysis results",
        "policies": "RL policies and decision rules",
        "ablation": "Ablation study results"
    }
    
    print(colored("\n📂 Creating project directory structure:", "cyan"))
    
    # Create main directory
    if os.path.exists(run_dir):
        print(colored(f"⚠️ Directory {run_dir} already exists", "yellow"))
    else:
        try:
            os.makedirs(run_dir, exist_ok=True)
            print(colored(f"✅ Created main directory: {run_dir}", "green"))
        except Exception as e:
            print(colored(f"❌ Error creating directory {run_dir}: {e}", "red"))
            sys.exit(1)
    
    # Create subdirectories with descriptions in a neat table
    print(colored("\n📁 Creating subdirectories:", "cyan"))
    print(colored("   ┌─────────────────┬─────────────────────────────────────┐", "cyan"))
    print(colored("   │ Directory       │ Description                         │", "cyan"))
    print(colored("   ├─────────────────┼─────────────────────────────────────┤", "cyan"))
    
    for subdir, description in subdirs.items():
        subdir_path = os.path.join(run_dir, subdir)
        try:
            os.makedirs(subdir_path, exist_ok=True)
            status = "✅"
            color = "green"
        except Exception as e:
            status = "❌"
            color = "red"
            print(colored(f"Error creating {subdir_path}: {e}", "red"))
        
        print(colored(f"   │ {status} {subdir.ljust(14)} │ {description.ljust(37)} │", color))
    
    print(colored("   └─────────────────┴─────────────────────────────────────┘", "cyan"))
    
    # Create README file in the run directory
    readme_path = os.path.join(run_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"# Process Mining Results - {timestamp}\n\n")
        f.write("This directory contains results from process mining analysis using GNN, LSTM, and RL techniques.\n\n")
        f.write("## Directory Structure\n\n")
        for subdir, description in subdirs.items():
            f.write(f"- **{subdir}**: {description}\n")
        f.write("\n## Runtime Information\n\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Time**: {datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"- **Command**: {' '.join(sys.argv)}\n")
    
    return run_dir

def save_metrics(metrics_dict, run_dir, filename, pretty=True):
    """
    Save metrics to JSON file with improved formatting
    
    Args:
        metrics_dict: Dictionary of metrics to save
        run_dir: Directory to save metrics
        filename: Name of metrics file
        pretty: Whether to format JSON for readability
    
    Returns:
        Path to saved metrics file
    """
    filepath = os.path.join(run_dir, "metrics", filename)
    
    try:
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(metrics_dict, f, indent=4, sort_keys=True, cls=NumpyEncoder)
            else:
                json.dump(metrics_dict, f, cls=NumpyEncoder)
        
        file_size = os.path.getsize(filepath)
        print(colored(f"✅ Saved metrics to {filename} ({file_size/1024:.1f} KB)", "green"))
        
    except Exception as e:
        print(colored(f"❌ Error saving metrics to {filename}: {e}", "red"))
    
    return filepath

def print_section_header(title, width=80):
    """
    Print a visually appealing section header
    
    Args:
        title: Header title 
        width: Width of header
    """
    print("\n" + "=" * width)
    print(colored(f" 🔍 {title}", "cyan", attrs=["bold"]))
    print("=" * width)

class ExperimentManager:
    """
    Experiment management with metrics tracking and artifact management
    """
    
    def __init__(self, args, run_dir):
        """
        Initialize experiment manager
        
        Args:
            args: Command-line arguments
            run_dir: Results directory
        """
        self.args = args
        self.run_dir = run_dir
        self.start_time = time.time()
        self.metrics = {}
        
        # Set up paths
        self.models_dir = os.path.join(run_dir, "models")
        self.metrics_dir = os.path.join(run_dir, "metrics")
        self.viz_dir = os.path.join(run_dir, "visualizations")
        self.analysis_dir = os.path.join(run_dir, "analysis")
        self.policies_dir = os.path.join(run_dir, "policies")
        self.ablation_dir = os.path.join(run_dir, "ablation")
        
        # Ensure directories exist
        for directory in [self.models_dir, self.metrics_dir, self.viz_dir, 
                          self.analysis_dir, self.policies_dir, self.ablation_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save initial arguments
        self._save_experiment_config()
    
    def _save_experiment_config(self):
        """Save experiment configuration as JSON"""
        try:
            # Convert args to dict if needed
            config = vars(self.args) if hasattr(self.args, '__dict__') else self.args
            
            # Save to file
            config_path = os.path.join(self.metrics_dir, "experiment_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4, sort_keys=True, cls=NumpyEncoder)
            
            print(colored(f"✅ Saved experiment configuration to experiment_config.json", "green"))
        except Exception as e:
            print(colored(f"❌ Error saving experiment configuration: {e}", "red"))
    
    def log_metrics(self, name, metrics_dict):
        """
        Log metrics for a component
        
        Args:
            name: Component name
            metrics_dict: Dictionary of metrics
        """
        self.metrics[name] = metrics_dict
        
        # Save individual metrics file
        save_metrics(metrics_dict, self.run_dir, f"{name}_metrics.json")
    
    def save_model(self, model, name):
        """
        Save a model to the models directory
        
        Args:
            model: Model to save
            name: Model name
        
        Returns:
            Path to saved model
        """
        import torch
        
        model_path = os.path.join(self.models_dir, f"{name}.pth")
        
        try:
            torch.save(model.state_dict(), model_path)
            print(colored(f"✅ Saved model to {name}.pth", "green"))
        except Exception as e:
            print(colored(f"❌ Error saving model: {e}", "red"))
        
        return model_path
    
    def generate_report(self):
        """
        Generate a comprehensive experiment report
        
        Returns:
            Path to report file
        """
        report_path = os.path.join(self.run_dir, "experiment_report.md")
        
        try:
            # Calculate duration
            duration = time.time() - self.start_time
            duration_str = f"{duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s"
            
            with open(report_path, 'w') as f:
                f.write(f"# Process Mining Experiment Report\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Duration:** {duration_str}\n\n")
                
                # Configuration
                f.write("## Configuration\n\n")
                f.write("```\n")
                for key, value in vars(self.args).items():
                    f.write(f"{key}: {value}\n")
                f.write("```\n\n")
                
                # Model metrics
                if 'model' in self.metrics:
                    f.write("## Model Performance\n\n")
                    f.write("```\n")
                    for key, value in self.metrics['model'].items():
                        f.write(f"{key}: {value}\n")
                    f.write("```\n\n")
                
                # Process analysis
                if 'process_analysis' in self.metrics:
                    f.write("## Process Analysis\n\n")
                    f.write("```\n")
                    for key, value in self.metrics['process_analysis'].items():
                        f.write(f"{key}: {value}\n")
                    f.write("```\n\n")
                
                # List of artifacts
                f.write("## Generated Artifacts\n\n")
                
                # Models
                model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
                if model_files:
                    f.write("### Models\n\n")
                    for model_file in model_files:
                        f.write(f"- {model_file}\n")
                    f.write("\n")
                
                # Visualizations
                viz_files = os.listdir(self.viz_dir)
                if viz_files:
                    f.write("### Visualizations\n\n")
                    for viz_file in viz_files:
                        f.write(f"- {viz_file}\n")
                    f.write("\n")
                
                # Analysis
                analysis_files = os.listdir(self.analysis_dir)
                if analysis_files:
                    f.write("### Analysis\n\n")
                    for analysis_file in analysis_files:
                        f.write(f"- {analysis_file}\n")
                    f.write("\n")
            
            print(colored(f"✅ Generated experiment report at {report_path}", "green"))
        except Exception as e:
            print(colored(f"❌ Error generating report: {e}", "red"))
        
        return report_path