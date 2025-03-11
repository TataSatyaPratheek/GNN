#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment management for process mining
Handles result directories, file saving, and metrics tracking
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from colorama import colored

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

class Experiment:
    """
    Manages experiment execution, directories, and result saving
    """
    def __init__(self, custom_dir=None):
        """
        Initialize experiment
        
        Args:
            custom_dir: Optional custom output directory
        """
        self.run_dir = self.setup_results_dir(custom_dir)
        self.viz_dir = os.path.join(self.run_dir, "visualizations")
        self.models_dir = os.path.join(self.run_dir, "models")
        self.analysis_dir = os.path.join(self.run_dir, "analysis")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.policy_dir = os.path.join(self.run_dir, "policies")
        self.ablation_dir = os.path.join(self.run_dir, "ablation")
        
        self.start_time = time.time()
    
    def setup_results_dir(self, custom_dir=None):
        """
        Create organized results directory structure with timestamp
        
        Args:
            custom_dir: Optional custom directory path
            
        Returns:
            Path to run directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use absolute path with optional custom directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "../..", "results")
        
        if custom_dir:
            if os.path.isabs(custom_dir):
                run_dir = custom_dir
            else:
                run_dir = os.path.join(script_dir, "../..", custom_dir)
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
    
    def save_metrics(self, metrics_dict, filename, pretty=True):
        """
        Save metrics to JSON file with improved formatting
        
        Args:
            metrics_dict: Dictionary of metrics to save
            filename: Filename to save to
            pretty: Whether to format JSON for readability
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.metrics_dir, filename)
        
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
    
    def generate_summary(self, dataset_info, model_results, analysis_results):
        """
        Generate execution summary
        
        Args:
            dataset_info: Dataset information dictionary
            model_results: Model results dictionary
            analysis_results: Analysis results dictionary
            
        Returns:
            Path to summary file
        """
        total_duration = time.time() - self.start_time
        
        # Create summary
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": total_duration,
            "total_duration_formatted": f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s",
            "dataset": dataset_info,
            "models": model_results,
            "process_analysis": analysis_results
        }
        
        # Save summary
        self.save_metrics(summary, "execution_summary.json")
        
        # Generate summary report in markdown format
        report_path = os.path.join(self.run_dir, "execution_summary.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# Process Mining Execution Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write(f"**Time:** {datetime.now().strftime('%H:%M:%S')}\n")
            f.write(f"**Duration:** {summary['total_duration_formatted']}\n\n")
            
            f.write(f"## Dataset Information\n\n")
            f.write(f"- **Filename:** {dataset_info['filename']}\n")
            f.write(f"- **Cases:** {dataset_info['cases']:,}\n")
            f.write(f"- **Events:** {dataset_info['events']:,}\n")
            f.write(f"- **Activities:** {dataset_info['activities']}\n")
            f.write(f"- **Resources:** {dataset_info['resources']}\n\n")
            
            f.write(f"## Model Performance\n\n")
            for model_name, model_info in model_results.items():
                if isinstance(model_info, dict) and 'accuracy' in model_info:
                    f.write(f"- **{model_name.replace('_', ' ').title()} Accuracy:** {model_info['accuracy']:.4f}\n")
            
            f.write(f"\n## Process Analysis\n\n")
            f.write(f"- **Significant Bottlenecks:** {analysis_results.get('bottlenecks', 0)}\n")
            f.write(f"- **Median Cycle Time:** {analysis_results.get('median_cycle_time', 0):.2f} hours\n")
            f.write(f"- **95th Percentile Cycle Time:** {analysis_results.get('p95_cycle_time', 0):.2f} hours\n\n")
            
            # Add sections for generated artifacts based on what was created
            f.write(f"## Generated Artifacts\n\n")
            artifacts = []
            
            if os.path.exists(self.models_dir) and os.listdir(self.models_dir):
                artifacts.append("Models")
            
            if os.path.exists(self.viz_dir) and os.listdir(self.viz_dir):
                artifacts.append("Visualizations")
            
            if os.path.exists(self.analysis_dir) and os.listdir(self.analysis_dir):
                artifacts.append("Analysis")
                
            if os.path.exists(self.policy_dir) and os.listdir(self.policy_dir):
                artifacts.append("Policies")
                
            if os.path.exists(self.ablation_dir) and os.listdir(self.ablation_dir):
                artifacts.append("Ablation")
            
            for artifact in artifacts:
                f.write(f"- **{artifact}**\n")
        
        return report_path

def print_section_header(title, width=80):
    """
    Print a visually appealing section header
    
    Args:
        title: Header title
        width: Width of the header
    """
    print("\n" + "=" * width)
    print(colored(f" 🔍 {title}", "cyan", attrs=["bold"]))
    print("=" * width)