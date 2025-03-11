#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bottleneck analysis utilities for process mining
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def analyze_bottlenecks(df, freq_threshold=5):
    """
    Analyze process bottlenecks based on waiting times between activities
    
    Args:
        df: Process data dataframe
        freq_threshold: Minimum frequency threshold for significant bottlenecks
        
    Returns:
        Tuple of (bottleneck_stats, significant_bottlenecks)
    """
    print("\n==== Analyzing Process Bottlenecks ====")
    start_time = time.time()
    
    df = df.copy()
    
    print("Computing transitions and waiting times...")
    # Add next task and timestamp
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

def analyze_cycle_times(df, viz_dir=None):
    """
    Analyze process cycle times
    
    Args:
        df: Process data dataframe
        viz_dir: Optional directory to save visualizations
        
    Returns:
        Tuple of (case_merged, long_cases, p95)
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
    
    # Create and save cycle time distribution visualization if directory provided
    if viz_dir:
        from processmine.visualization.process_viz import plot_cycle_time_distribution
        plot_cycle_time_distribution(case_merged["duration_h"].values, 
                                    os.path.join(viz_dir, 'cycle_time_distribution.png'))
    
    print(f"Analysis completed in \033[96m{time.time() - start_time:.2f}s\033[0m")
    
    return case_merged, long_cases, p95

def analyze_rare_transitions(bottleneck_stats, rare_threshold=2):
    """
    Identify rare transitions in the process
    
    Args:
        bottleneck_stats: Bottleneck statistics dataframe
        rare_threshold: Maximum frequency threshold for rare transitions
        
    Returns:
        Dataframe of rare transitions
    """
    rare_trans = bottleneck_stats[bottleneck_stats["count"] <= rare_threshold]
    return rare_trans