# src/processmine/visualization/plotting.py
"""
General plotting utilities for data visualization.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set better style defaults for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
sns.set_palette("viridis")

def plot_cycle_time_distribution(durations, save_path="cycle_time_distribution.png"):
    """
    Plot enhanced cycle time distribution with better visuals and percentiles.
    
    Args:
        durations: Array of case durations in hours
        save_path: Path to save the distribution plot
        
    Returns:
        Dictionary with key statistics
    """
    print("\n==== Creating Cycle Time Distribution ====")
    start_time = time.time()
    
    # Calculate statistics
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    p90 = np.percentile(durations, 90)
    p95 = np.percentile(durations, 95)
    
    # Create figure with improved aesthetics
    plt.figure(figsize=(12, 7))
    
    # Plot histogram with KDE
    sns.histplot(durations, bins=min(50, len(durations) // 20), 
                kde=True, color="royalblue", edgecolor="white", alpha=0.7)
    
    # Add percentile lines
    plt.axvline(mean_duration, color="red", linestyle="-", 
               linewidth=2, label=f"Mean: {mean_duration:.1f}h")
    plt.axvline(median_duration, color="green", linestyle="--", 
               linewidth=2, label=f"Median: {median_duration:.1f}h")
    plt.axvline(p90, color="purple", linestyle="-.", 
               linewidth=2, label=f"90th Percentile: {p90:.1f}h")
    plt.axvline(p95, color="orange", linestyle="-.", 
               linewidth=2, label=f"95th Percentile: {p95:.1f}h")
    
    # Enhanced styling
    plt.title("Process Cycle Time Distribution", fontsize=16)
    plt.xlabel("Duration (hours)", fontsize=14)
    plt.ylabel("Number of Cases", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    
    # Add summary statistics as text
    stats_text = f"Total Cases: {len(durations)}\n"
    stats_text += f"Mean: {mean_duration:.2f}h\n"
    stats_text += f"Median: {median_duration:.2f}h\n"
    stats_text += f"Min: {np.min(durations):.2f}h\n"
    stats_text += f"Max: {np.max(durations):.2f}h\n"
    stats_text += f"Std Dev: {np.std(durations):.2f}h"
    
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Cycle time distribution saved to {save_path} in {time.time() - start_time:.2f}s")
    return {"mean": mean_duration, "median": median_duration, "p90": p90, "p95": p95}

def plot_metric_evolution(metrics, title="Model Training Progress", save_path=None):
    """
    Plot metrics evolution over epochs during model training.
    
    Args:
        metrics: Dictionary of lists with metrics evolution
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        The figure if save_path is None, otherwise None
    """
    plt.figure(figsize=(12, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(range(1, len(values) + 1), values, 
                marker='o', linestyle='-', label=metric_name)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return None
    else:
        return plt.gcf()

def plot_feature_importance(feature_names, importance_values, top_n=20, save_path=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of feature importance values
        top_n: Number of top features to display
        save_path: Path to save the plot
        
    Returns:
        DataFrame with feature importance sorted by value
    """
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    # Select top N features
    if top_n is not None and top_n < len(importance_df):
        plot_df = importance_df.head(top_n).copy()
    else:
        plot_df = importance_df.copy()
    
    # Create plot
    plt.figure(figsize=(12, max(6, len(plot_df) * 0.3)))
    
    # Plot horizontal bar chart
    ax = sns.barplot(x='Importance', y='Feature', data=plot_df, palette="viridis")
    
    # Add values to bars
    for i, v in enumerate(plot_df['Importance']):
        ax.text(v + 0.001, i, f"{v:.4f}", va='center')
    
    plt.title("Feature Importance", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    return importance_df