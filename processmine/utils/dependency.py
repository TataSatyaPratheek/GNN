#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dependency manager for process mining - handles optional dependencies gracefully
"""

import importlib
import logging
import functools
import warnings
from typing import Dict, List, Optional, Callable, Any, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Registry of dependency availability
_DEPENDENCY_REGISTRY = {}

# Registry of fallback functions
_FALLBACK_REGISTRY = {}

def check_dependency(package_name: str) -> bool:
    """
    Check if an optional dependency is installed
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if the package is available, False otherwise
    """
    # Check from registry if we've already checked this package
    if package_name in _DEPENDENCY_REGISTRY:
        return _DEPENDENCY_REGISTRY[package_name]
    
    # Try to import the package
    try:
        importlib.import_module(package_name)
        _DEPENDENCY_REGISTRY[package_name] = True
        return True
    except ImportError:
        _DEPENDENCY_REGISTRY[package_name] = False
        return False

def requires_dependency(package_name: str, error_message: Optional[str] = None):
    """
    Decorator for functions that require an optional dependency
    
    Args:
        package_name: Name of the package required
        error_message: Optional custom error message
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if check_dependency(package_name):
                return func(*args, **kwargs)
            else:
                # Use custom message or generate a default one
                msg = error_message
                if msg is None:
                    msg = f"Function '{func.__name__}' requires the package '{package_name}', which is not installed."
                    msg += f" Please install it with 'pip install {package_name}'."
                
                # Check if there's a fallback function registered
                fallback_key = f"{func.__module__}.{func.__name__}"
                if fallback_key in _FALLBACK_REGISTRY:
                    logger.warning(f"{msg} Using fallback implementation.")
                    fallback_func = _FALLBACK_REGISTRY[fallback_key]
                    return fallback_func(*args, **kwargs)
                
                # No fallback, raise error
                raise ImportError(msg)
        
        # Store the original unwrapped function for inspection
        wrapper._original = func
        return wrapper
    
    return decorator

def register_fallback(original_func: Callable, fallback_func: Callable) -> None:
    """
    Register a fallback function for an optional dependency
    
    Args:
        original_func: Original function that requires dependencies
        fallback_func: Fallback function to use if dependencies are missing
    """
    key = f"{original_func.__module__}.{original_func.__name__}"
    _FALLBACK_REGISTRY[key] = fallback_func
    logger.debug(f"Registered fallback for {key}")

def with_fallback(package_name: str, fallback_func: Callable):
    """
    Decorator to specify a fallback function for optional dependencies
    
    Args:
        package_name: Name of the required package
        fallback_func: Fallback function to use if the package is missing
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if check_dependency(package_name):
                return func(*args, **kwargs)
            else:
                warning_msg = f"Package '{package_name}' not available. Using fallback for '{func.__name__}'."
                warnings.warn(warning_msg)
                logger.warning(warning_msg)
                return fallback_func(*args, **kwargs)
        
        return wrapper
    
    return decorator

# Specific dependency checks for common optional packages
HAVE_PLOTLY = check_dependency("plotly")
HAVE_NETWORKX = check_dependency("networkx")
HAVE_UMAP = check_dependency("umap")
HAVE_PM4PY = check_dependency("pm4py")
HAVE_TORCH_GEOMETRIC = check_dependency("torch_geometric")
HAVE_XGB = check_dependency("xgboost")

def get_dependency_status() -> Dict[str, bool]:
    """
    Get the status of all checked dependencies
    
    Returns:
        Dictionary mapping package names to availability status
    """
    # Add common packages if not already checked
    for pkg in ["torch", "numpy", "pandas", "matplotlib", 
                "seaborn", "sklearn", "tqdm", "plotly", 
                "networkx", "umap", "pm4py", "torch_geometric",
                "xgboost"]:
        if pkg not in _DEPENDENCY_REGISTRY:
            check_dependency(pkg)
    
    return _DEPENDENCY_REGISTRY.copy()

def print_dependency_status() -> None:
    """Print the status of all dependencies in a nice format"""
    status = get_dependency_status()
    
    # Get max package name length for formatting
    max_len = max(len(pkg) for pkg in status.keys())
    
    print("\n=== Dependency Status ===")
    print(f"{'Package':<{max_len+2}} | Status")
    print("-" * (max_len + 12))
    
    # Sort by name
    for pkg, available in sorted(status.items()):
        status_str = "✅ Available" if available else "❌ Missing"
        print(f"{pkg:<{max_len+2}} | {status_str}")

def get_missing_dependencies() -> List[str]:
    """
    Get a list of missing dependencies
    
    Returns:
        List of missing package names
    """
    status = get_dependency_status()
    return [pkg for pkg, available in status.items() if not available]

def generate_install_command() -> str:
    """
    Generate a pip install command for missing dependencies
    
    Returns:
        Pip install command string
    """
    missing = get_missing_dependencies()
    if not missing:
        return "# All dependencies are installed"
    
    return f"pip install {' '.join(missing)}"

# Define fallbacks for common functionalities

# Simple fallback sankey diagram using matplotlib instead of plotly
def fallback_sankey_diagram(transitions, le_task, save_path, **kwargs):
    """
    Fallback Sankey diagram using matplotlib instead of plotly
    
    Args:
        transitions: DataFrame with transitions data
        le_task: Task label encoder
        save_path: Path to save the diagram
        **kwargs: Additional arguments (ignored)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.warning("Using simplified Sankey diagram fallback (plotly not available)")
    
    # Extract case-level information from transitions
    df = transitions.copy()
    
    # Calculate transition counts
    trans_count = df.groupby(["task_id", "next_task_id"]).size().reset_index(name='count')
    
    # Create node list
    unique_tasks = sorted(set(df["task_id"].unique()).union(set(df["next_task_id"].dropna().unique())))
    
    try:
        # Map task IDs to readable names
        task_names = {task_id: le_task.inverse_transform([int(task_id)])[0] for task_id in unique_tasks}
    except:
        # Fallback if transformation fails
        task_names = {task_id: f"Task {task_id}" for task_id in unique_tasks}
    
    # Create simplified flow diagram
    plt.figure(figsize=(12, 8))
    
    # Create a directed graph visualization
    import matplotlib.lines as mlines
    
    # Place nodes in a circle
    n_nodes = len(unique_tasks)
    radius = 5
    node_positions = {}
    
    for i, task_id in enumerate(unique_tasks):
        angle = 2 * np.pi * i / n_nodes
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        node_positions[task_id] = (x, y)
    
    # Draw edges
    max_count = trans_count['count'].max()
    
    for _, row in trans_count.iterrows():
        src_id, tgt_id, count = row["task_id"], row["next_task_id"], row["count"]
        if pd.notna(tgt_id):  # Skip NaN targets
            src_pos = node_positions[src_id]
            tgt_pos = node_positions[tgt_id]
            
            # Edge width based on count
            width = 0.5 + 5.0 * (count / max_count)
            
            # Draw arrow
            arrow = mlines.Line2D(
                [src_pos[0], tgt_pos[0]], 
                [src_pos[1], tgt_pos[1]],
                linewidth=width,
                color='blue',
                alpha=0.5,
                zorder=1
            )
            plt.gca().add_line(arrow)
    
    # Draw nodes
    for task_id, (x, y) in node_positions.items():
        plt.scatter(x, y, s=300, color='skyblue', edgecolor='black', zorder=2)
        plt.text(x, y, task_names[task_id], ha='center', va='center', fontsize=8)
    
    plt.title("Process Flow Diagram (Simple Fallback)")
    plt.axis('equal')
    plt.axis('off')
    
    # Save as PNG instead of HTML
    png_path = save_path.replace(".html", ".png")
    plt.savefig(png_path)
    plt.close()
    
    print(f"Simplified flow diagram saved to {png_path}")
    
    return None

# Register fallbacks for common functions
try:
    from processmine.visualization.process_viz import create_sankey_diagram
    register_fallback(create_sankey_diagram, fallback_sankey_diagram)
except ImportError:
    pass

# Fallback for dimensionality reduction without UMAP
def fallback_dimensionality_reduction(embeddings, method="tsne", **kwargs):
    """
    Fallback for dimensionality reduction without UMAP
    
    Args:
        embeddings: High-dimensional embeddings
        method: Method to use (will use t-SNE regardless)
        **kwargs: Additional arguments
        
    Returns:
        2D coordinates
    """
    from sklearn.manifold import TSNE
    
    logger.warning("UMAP not available. Falling back to t-SNE for dimensionality reduction.")
    
    # Ensure perplexity is less than number of samples
    safe_perplexity = min(30, max(5, embeddings.shape[0] - 1))
    logger.info(f"Running t-SNE with perplexity {safe_perplexity}...")
    
    coords = TSNE(
        n_components=2, 
        perplexity=safe_perplexity, 
        random_state=42, 
        n_iter=1000
    ).fit_transform(embeddings)
    
    return coords