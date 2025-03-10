#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Analysis Module
Includes bottleneck analysis, conformance checking, and cycle time analysis
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os

# Optional PM4Py import with fallback
try:
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("\033[93mWarning: PM4Py not available. Some process mining functions will be limited.\033[0m")

def analyze_bottlenecks(df, freq_threshold=5):
    """
    Analyze process bottlenecks based on waiting times between activities
    """
    print("\n==== Analyzing Process Bottlenecks ====")
    start_time = time.time()
    
    df = df.copy()
    
    print("Computing transitions and waiting times...")
    # Add next task and timestamp with progress feedback
    df["next_task_id"] = df.groupby("case_id")["task_id"].shift(-1)
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    
    # Filter transitions
    transitions = df.dropna(subset=["next_task_id"]).copy()
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    print("Analyzing waiting times...")
    # Group and calculate statistics
    bottleneck_stats = transitions.groupby(["task_id","next_task_id"])["wait_sec"].agg([
        "mean","median","std","count"
    ]).reset_index()
    
    # Add derived metrics
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"]/3600.0
    bottleneck_stats["coefficient_variation"] = bottleneck_stats["std"] / bottleneck_stats["mean"]
    
    # Sort by average wait time
    bottleneck_stats.sort_values("mean_hours", ascending=False, inplace=True)
    
    # Filter by frequency threshold
    significant_bottlenecks = bottleneck_stats[bottleneck_stats["count"] >= freq_threshold]
    
    # Print summary
    top_bottlenecks = significant_bottlenecks.head(5)
    print("\033[1mTop Bottlenecks\033[0m:")
    for i, row in top_bottlenecks.iterrows():
        print(f"  Task {int(row['task_id'])} → Task {int(row['next_task_id'])}: "
              f"\033[93m{row['mean_hours']:.2f} hours\033[0m avg wait "
              f"({int(row['count'])} occurrences)")
    
    print(f"Analyzed {len(transitions):,} transitions, identified {len(significant_bottlenecks)} significant bottlenecks")
    print(f"Analysis completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return bottleneck_stats, significant_bottlenecks

def analyze_cycle_times(df):
    """
    Analyze process cycle times with enhanced progress tracking
    """
    print("\n==== Analyzing Cycle Times ====")
    start_time = time.time()
    
    print("Computing case durations...")
    # Group by case and calculate min/max timestamps
    case_grouped = df.groupby("case_id")["timestamp"].agg(["min","max"])
    case_grouped["cycle_time_hours"] = (
        case_grouped["max"] - case_grouped["min"]
    ).dt.total_seconds()/3600.0
    case_grouped.reset_index(inplace=True)
    
    print("Computing case attributes...")
    # Add case features
    df_feats = df.groupby("case_id").agg({
        "amount": "mean",
        "task_id": "count"
    }).rename(columns={
        "amount": "mean_amount",
        "task_id": "num_events"
    }).reset_index()
    
    # Merge features
    case_merged = pd.merge(case_grouped, df_feats, on="case_id", how="left")
    case_merged["duration_h"] = case_merged["cycle_time_hours"]
    
    # Calculate percentiles
    p50 = case_merged["duration_h"].median()
    p95 = case_merged["duration_h"].quantile(0.95)
    p99 = case_merged["duration_h"].quantile(0.99)
    max_duration = case_merged["duration_h"].max()
    
    # Identify long-running cases (95th percentile)
    long_cases = case_merged[case_merged["duration_h"] > p95]
    
    # Analyze correlation with case attributes
    corr_events = case_merged["duration_h"].corr(case_merged["num_events"])
    corr_amount = case_merged["duration_h"].corr(case_merged["mean_amount"])
    
    # Print summary
    print("\033[1mCycle Time Statistics\033[0m:")
    print(f"  Median (P50): \033[96m{p50:.2f} hours\033[0m")
    print(f"  95th Percentile: \033[93m{p95:.2f} hours\033[0m")
    print(f"  99th Percentile: \033[91m{p99:.2f} hours\033[0m")
    print(f"  Maximum: \033[91m{max_duration:.2f} hours\033[0m")
    print(f"  Long-running cases: {len(long_cases)} (>95th percentile)")
    
    print("\033[1mCorrelations\033[0m:")
    print(f"  Duration vs. Events: {corr_events:.2f}")
    print(f"  Duration vs. Amount: {corr_amount:.2f}")
    
    print(f"Analysis completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return case_merged, long_cases, p95

def analyze_rare_transitions(bottleneck_stats, rare_threshold=2):
    """
    Identify rare transitions in the process
    """
    rare_trans = bottleneck_stats[bottleneck_stats["count"] <= rare_threshold]
    return rare_trans

def perform_conformance_checking(df):
    """
    Perform conformance checking using inductive miner and token replay
    """
    if not PM4PY_AVAILABLE:
        print("\033[93mPM4Py not available. Skipping conformance checking.\033[0m")
        return [], 0
    
    print("\n==== Performing Conformance Checking ====")
    start_time = time.time()
    
    print("Preparing event log...")
    # Prepare dataframe for PM4Py
    df_pm = df[["case_id","task_name","timestamp"]].rename(columns={
        "case_id": "case:concept:name",
        "task_name": "concept:name",
        "timestamp": "time:timestamp"
    })
    
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)
    
    print("Converting to event log format...")
    event_log = log_converter.apply(df_pm)
    
    print("Discovering process model...")
    process_tree = inductive_miner.apply(event_log)
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    net, im, fm = pt_converter.apply(process_tree)
    
    print("Performing token replay...")
    # Add progress bar for token replay if possible
    try:
        from pm4py.objects.conversion.process_tree import converter as pt_converter
        replayed = token_replay.apply(event_log, net, im, fm)
    except Exception as e:
        print(f"\033[91mError during token replay: {e}\033[0m")
        print("Falling back to simplified conformance checking...")
        # Simple fallback - just count variants
        variants = df.groupby("case_id")["task_name"].agg(lambda x: tuple(x)).value_counts()
        top_variant_count = variants.iloc[0]
        total_cases = len(variants)
        conformance = top_variant_count / total_cases
        print(f"Top variant covers {conformance:.1%} of cases ({top_variant_count}/{total_cases})")
        return [], 0
    
    # Count non-conforming traces
    n_deviant = sum(1 for t in replayed if not t["trace_is_fit"])
    fit_percentage = (len(replayed) - n_deviant) / len(replayed) * 100 if replayed else 0
    
    print(f"\033[1mConformance Results\033[0m:")
    print(f"  Model-conforming traces: \033[96m{len(replayed) - n_deviant}\033[0m ({fit_percentage:.1f}%)")
    print(f"  Deviant traces: \033[93m{n_deviant}\033[0m ({100-fit_percentage:.1f}%)")
    print(f"  Total traces: {len(replayed)}")
    print(f"Analysis completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return replayed, n_deviant

def analyze_transition_patterns(df):
    """
    Analyze transition patterns and compute transition matrix
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Limit to top activities for readability
        top_n = 10
        top_tasks = trans_count.sum(axis=1).nlargest(top_n).index
        top_matrix = prob_matrix.loc[top_tasks, top_tasks]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(top_matrix, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title(f'Transition Probability Heatmap (Top {top_n} Activities)')
        plt.xlabel('Next Activity')
        plt.ylabel('Current Activity')
        plt.tight_layout()
        plt.savefig('transition_heatmap.png')
        print("Saved transition heatmap to transition_heatmap.png")
    except:
        pass
    
    return transitions, trans_count, prob_matrix

def spectral_cluster_graph(adj_matrix, k=2):
    """
    Perform spectral clustering on process graph with improved implementation
    """
    print("\n==== Performing Spectral Clustering ====")
    start_time = time.time()
    
    from sklearn.cluster import KMeans
    import numpy as np
    
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