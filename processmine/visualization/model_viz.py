# src/processmine/visualization/model_viz.py
"""
Model visualization tools for model analysis and evaluation.
"""

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE

# Make UMAP optional to avoid dependency issues
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("Warning: UMAP not available. Dimensionality reduction will use t-SNE only.")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """
    Plot enhanced confusion matrix with improved visuals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        save_path: Path to save the confusion matrix image
        
    Returns:
        Tuple of (accuracy, f1_score)
    """
    print("\n==== Creating Confusion Matrix ====")
    start_time = time.time()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for better interpretation
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Handle potential NaN values
    cm_norm = np.nan_to_num(cm_norm)
    
    # Calculate plot size based on class count
    n_classes = len(class_names)
    fig_size = max(8, min(20, n_classes / 2))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
    
    # Plot absolute confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names if n_classes <= 20 else [],
                yticklabels=class_names if n_classes <= 20 else [],
                ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (Absolute Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names if n_classes <= 20 else [],
                yticklabels=class_names if n_classes <= 20 else [],
                ax=axes[1], cbar=True)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    # If many classes, rotate labels
    if n_classes > 10:
        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=90)
            plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Add overall metrics
    plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.suptitle("Model Performance Analysis", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save with high quality
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path} in {time.time() - start_time:.2f}s")
    return accuracy, f1

def plot_embeddings(embeddings, labels=None, method="tsne", save_path=None):
    """
    Plot task embeddings using t-SNE or UMAP with enhanced visuals.
    
    Args:
        embeddings: Embeddings array [n_samples, n_features]
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ('tsne' or 'umap')
        save_path: Path to save the visualization
        
    Returns:
        Reduced dimensions coordinates
    """
    print(f"\n==== Creating Embeddings Visualization (using {method}) ====")
    start_time = time.time()
    
    # Handle UMAP not being available
    if method.lower() == "umap" and not UMAP_AVAILABLE:
        print("Warning: UMAP not available. Falling back to t-SNE.")
        method = "tsne"
    
    # Calculate embeddings
    if method.lower() == "tsne":
        # Ensure perplexity is less than number of samples
        safe_perplexity = min(30, max(5, embeddings.shape[0] - 1))
        print(f"Running t-SNE with perplexity {safe_perplexity}...")
        coords = TSNE(n_components=2, perplexity=safe_perplexity, 
                      random_state=42, n_jobs=-1).fit_transform(embeddings)
        title = "Task Embeddings - t-SNE"
    else:  # umap
        print("Running UMAP...")
        try:
            # Ensure n_neighbors is less than number of samples
            n_neighbors = min(30, max(2, embeddings.shape[0] // 2))
            coords = umap.UMAP(n_components=2, random_state=42,
                               n_neighbors=n_neighbors,
                               min_dist=0.1).fit_transform(embeddings)
            title = "Task Embeddings - UMAP"
        except Exception as e:
            print(f"Error running UMAP: {e}")
            print("Falling back to t-SNE...")
            # Safe perplexity for t-SNE
            safe_perplexity = min(30, max(5, embeddings.shape[0] - 1))
            coords = TSNE(n_components=2, perplexity=safe_perplexity, 
                          random_state=42, n_jobs=-1).fit_transform(embeddings)
            title = "Task Embeddings - t-SNE (UMAP failed)"
                
    # Create plot with better visuals
    plt.figure(figsize=(10, 8))
    
    # If labels are provided, use them for coloring
    if labels is not None:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', 
                              alpha=0.8, s=100, edgecolors='w', linewidths=0.5)
        plt.colorbar(scatter, label="Task ID" if isinstance(labels[0], (int, float)) else "Cluster")
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.8, s=100, 
                    edgecolors='w', linewidths=0.5)
    
    # Add annotations if not too many points
    if embeddings.shape[0] <= 30 and labels is not None:
        for i, (x, y) in enumerate(coords):
            plt.annotate(str(labels[i]), (x, y), fontsize=9, 
                        ha='center', va='center', color='black', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add timestamp and method info
    plt.figtext(0.02, 0.02, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", fontsize=8)
    plt.figtext(0.98, 0.02, f"Method: {method.upper()}", fontsize=8, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Embeddings visualization saved to {save_path} in {time.time() - start_time:.2f}s")
    
    plt.close()
    return coords