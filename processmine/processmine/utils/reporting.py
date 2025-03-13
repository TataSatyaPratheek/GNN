"""
Simplified reporting utilities for process mining results
"""
import os
import json
import time
from datetime import datetime
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_json(data: Dict[str, Any], filepath: str, pretty: bool = True) -> str:
    """
    Save data to JSON file with proper encoding
    
    Args:
        data: Data to save
        filepath: Path to save file
        pretty: Whether to use pretty formatting
        
    Returns:
        Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
            else:
                json.dump(data, f, cls=NumpyEncoder)
        return filepath
    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")
        return ""

def generate_report(args: Any, df: Any, model_type: str, 
                  metrics: Optional[Dict[str, Any]] = None,
                  output_dir: Optional[str] = None) -> str:
    """
    Generate final report with results summary
    
    Args:
        args: Command-line arguments
        df: Processed dataframe
        model_type: Type of model used
        metrics: Model evaluation metrics
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    # Skip if no output directory
    if not output_dir:
        return ""
    
    # Create report data
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time": time.time(),
        "model": {
            "type": model_type,
            "metrics": metrics or {}
        },
        "data": {
            "source": getattr(args, "data_path", "unknown"),
            "cases": df["case_id"].nunique(),
            "events": len(df),
            "activities": df["task_id"].nunique(),
            "resources": df["resource_id"].nunique(),
            "time_period": [
                df["timestamp"].min().strftime("%Y-%m-%d"),
                df["timestamp"].max().strftime("%Y-%m-%d")
            ]
        },
        "config": vars(args) if hasattr(args, "__dict__") else {}
    }
    
    # Save JSON report
    json_path = os.path.join(output_dir, "report.json")
    save_json(report, json_path)
    
    # Create markdown report
    md_path = os.path.join(output_dir, "report.md")
    
    with open(md_path, 'w') as f:
        f.write(f"# Process Mining Report\n\n")
        f.write(f"Generated: {report['timestamp']}\n\n")
        
        f.write(f"## Dataset Information\n\n")
        f.write(f"- **Source**: {report['data']['source']}\n")
        f.write(f"- **Cases**: {report['data']['cases']:,}\n")
        f.write(f"- **Events**: {report['data']['events']:,}\n")
        f.write(f"- **Activities**: {report['data']['activities']}\n")
        f.write(f"- **Resources**: {report['data']['resources']}\n")
        f.write(f"- **Time Period**: {report['data']['time_period'][0]} to {report['data']['time_period'][1]}\n\n")
        
        f.write(f"## Model Performance\n\n")
        if metrics:
            f.write(f"- **Model Type**: {model_type}\n")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"- **{metric}**: {value:.4f}\n")
            f.write("\n")
        else:
            f.write("No model metrics available.\n\n")
        
        f.write(f"## Generated Artifacts\n\n")
        
        # Find artifacts by extension
        artifacts = []
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.png', '.html', '.csv', '.json', '.pt')):
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    artifacts.append(rel_path)
        
        # Group artifacts by type
        artifact_groups = {}
        for artifact in artifacts:
            ext = os.path.splitext(artifact)[1]
            if ext not in artifact_groups:
                artifact_groups[ext] = []
            artifact_groups[ext].append(artifact)
        
        # List artifacts by type
        for ext, files in artifact_groups.items():
            f.write(f"### {ext[1:].upper()} Files\n\n")
            for file in sorted(files):
                f.write(f"- {file}\n")
            f.write("\n")
    
    logger.info(f"Generated report: {md_path}")
    return md_path