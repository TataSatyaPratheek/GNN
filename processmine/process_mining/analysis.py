#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process mining analysis utilities
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

def analyze_transition_patterns(df, viz_dir=None):
    """
    Analyze transition patterns and compute transition matrix
    
    Args:
        df: Process data dataframe
        viz_dir: Optional directory to save visualizations
        
    Returns:
        Tuple of (transitions, transition_count, probability_matrix)
    """
    print("\n==== Analyzing Transition Patterns ====")
    start_time = time.time()
    
    print("Computing transitions...")
    transitions = df.copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    
    # Filter out last events in cases
    transitions = transitions.dropna(subset=["next_task_id"])
    
    print("Building transition matrices...")
    # Create transition count matrix and probability matrix
    trans_count = transitions.groupby(["task_id","next_task_id"]).size().unstack(fill_value=0)
    prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
    
    # Analyze transition statistics
    total_transitions = trans_count.sum().sum()
    unique_transitions = (trans_count > 0).sum().sum()
    max_outgoing = trans_count.sum(axis=1).max()
    max_incoming = trans_count.sum(axis=0).max()
    
    print("\033[1mTransition Statistics\033[0m:")
    print(f"  Total transitions: \033[96m{total_transitions:,}\033[0m")
    print(f"  Unique transitions: \033[96m{unique_transitions}\033[0m")
    print(f"  Max outgoing transitions from a task: \033[96m{max_outgoing}\033[0m")
    print(f"  Max incoming transitions to a task: \033[96m{max_incoming}\033[0m")
    print(f"Analysis completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    # Plot transition heatmap if matplotlib is available
    try:
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Limit to top activities for readability
        top_n = 10
        top_tasks = trans_count.sum(axis=1).nlargest(top_n).index
        top_matrix = prob_matrix.loc[top_tasks, top_tasks]
        
        # Create heatmap
        sns.heatmap(top_matrix, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title(f'Transition Probability Heatmap (Top {top_n} Activities)')
        plt.xlabel('Next Activity')
        plt.ylabel('Current Activity')
        plt.tight_layout()
        
        # Save to visualization directory if provided
        if viz_dir:
            heatmap_path = os.path.join(viz_dir, 'transition_heatmap.png')
            plt.savefig(heatmap_path)
            print(f"Saved transition heatmap to {heatmap_path}")
        else:
            plt.savefig('transition_heatmap.png')
            print("Saved transition heatmap to transition_heatmap.png")
        
        plt.close()
    except Exception as e:
        print(f"Error creating transition heatmap: {e}")
    
    return transitions, trans_count, prob_matrix

def spectral_cluster_graph(adj_matrix, k=2):
    """
    Perform spectral clustering on process graph
    
    Args:
        adj_matrix: Adjacency matrix of the process graph
        k: Number of clusters
        
    Returns:
        Cluster labels for each node
    """
    print("\n==== Performing Spectral Clustering ====")
    start_time = time.time()
    
    # Make matrix symmetric to ensure real eigenvalues
    adj_symmetric = (adj_matrix + adj_matrix.T) / 2
    
    # Compute degree matrix
    degrees = np.sum(adj_symmetric, axis=1)
    D = np.diag(degrees)
    
    # Compute Laplacian (unnormalized)
    L = D - adj_symmetric
    
    print("Computing eigenvalues and eigenvectors...")
    # Compute eigenvectors efficiently
    try:
        # Use scipy.sparse.linalg for large matrices
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        
        if adj_matrix.shape[0] > 100:  # For large matrices
            L_sparse = sp.csr_matrix(L)
            eigenvals, eigenvecs = eigsh(L_sparse, k=k+1, which='SM')
        else:
            eigenvals, eigenvecs = np.linalg.eigh(L)
    except:
        # Fallback to standard numpy
        eigenvals, eigenvecs = np.linalg.eigh(L)
    
    # Sort eigenvectors by eigenvalues
    idx = np.argsort(eigenvals)
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
    
    # Calculate eigenvalue gaps to validate k
    if len(eigenvals) > k+1:
        gaps = np.diff(eigenvals[:k+2])
        best_gap_idx = np.argmax(gaps[:k])
        suggested_k = best_gap_idx + 1
        
        if suggested_k != k:
            print(f"\033[93mNote: Based on eigenvalue gaps, k={suggested_k} might be more appropriate than k={k}\033[0m")
    
    print(f"Using clustering with k={k}...")
    if k == 2:
        # Fiedler vector = second smallest eigenvector
        fiedler_vec = np.real(eigenvecs[:, 1])
        # Partition by sign
        labels = (fiedler_vec >= 0).astype(int)
    else:
        # multi-cluster
        embedding = np.real(eigenvecs[:, 1:k+1])
        
        # Normalize rows to unit length
        row_norms = np.sqrt(np.sum(embedding**2, axis=1))
        embedding = embedding / (row_norms[:, np.newaxis] + 1e-10)
        
        # Use multiple initializations for more stable clustering
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embedding)
        labels = kmeans.labels_
    
    # Compute cluster statistics
    cluster_sizes = np.bincount(labels)
    
    print("\033[1mClustering Results\033[0m:")
    for i in range(len(cluster_sizes)):
        print(f"  Cluster {i}: \033[96m{cluster_sizes[i]}\033[0m tasks")
    
    # Check for imbalanced clustering
    if max(cluster_sizes) > 0.9 * sum(cluster_sizes):
        print("\033[93mWarning: Highly imbalanced clustering detected. Consider different k value.\033[0m")
    
    print(f"Clustering completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return labels

def build_task_adjacency(df, num_tasks):
    """
    Build adjacency matrix weighted by transition frequencies
    
    Args:
        df: Process data dataframe
        num_tasks: Number of tasks
        
    Returns:
        Weighted adjacency matrix
    """
    print("\n==== Building Task Adjacency Matrix ====")
    start_time = time.time()
    
    # Initialize matrix
    A = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    
    # Group by case for more efficient processing
    case_groups = df.groupby("case_id")
    
    # Create progress bar
    progress_bar = tqdm(
        case_groups, 
        desc="Processing cases",
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=100
    )
    
    # Build adjacency matrix
    for cid, cdata in progress_bar:
        cdata = cdata.sort_values("timestamp")
        tasks_seq = cdata["task_id"].values
        for i in range(len(tasks_seq)-1):
            src = int(tasks_seq[i])
            tgt = int(tasks_seq[i+1])
            A[src, tgt] += 1.0
    
    # Add reverse edges for undirected clustering
    A_sym = A + A.T
    
    # Analyze matrix
    non_zero = np.count_nonzero(A)
    max_weight = np.max(A)
    total_weight = np.sum(A)
    density = non_zero / (num_tasks * num_tasks)
    
    print("\033[1mAdjacency Matrix Statistics\033[0m:")
    print(f"  Matrix shape: \033[96m{A.shape}\033[0m")
    print(f"  Non-zero entries: \033[96m{non_zero}\033[0m ({density:.1%} density)")
    print(f"  Max edge weight: \033[96m{max_weight:.1f}\033[0m")
    print(f"  Total edge weight: \033[96m{total_weight:.1f}\033[0m")
    print(f"Matrix built in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return A_sym