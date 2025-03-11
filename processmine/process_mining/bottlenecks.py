#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized bottleneck analysis for process mining with vectorized operations
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import logging
from typing import Tuple, Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

def analyze_bottlenecks_vectorized(df: pd.DataFrame, 
                                  freq_threshold: int = 5,
                                  percentile_threshold: float = 90.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze process bottlenecks using vectorized operations
    
    Args:
        df: Process data dataframe
        freq_threshold: Minimum frequency threshold for significant bottlenecks
        percentile_threshold: Percentile threshold for identifying bottlenecks
        
    Returns:
        Tuple of (bottleneck_stats, significant_bottlenecks)
    """
    logger.info("Analyzing process bottlenecks with vectorized operations")
    start_time = time.time()
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Efficiently add next_task_id and next_timestamp columns
    logger.debug("Computing transitions and wait times")
    df_copy["next_task_id"] = df_copy.groupby("case_id")["task_id"].shift(-1)
    df_copy["next_timestamp"] = df_copy.groupby("case_id")["timestamp"].shift(-1)
    
    # Filter transitions in one step
    transitions = df_copy.dropna(subset=["next_task_id"]).copy()
    
    # Vectorized computation of wait times
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    # Group and calculate statistics using optimized pandas operations
    logger.debug("Calculating transition statistics")
    bottleneck_stats = transitions.groupby(["task_id", "next_task_id"])["wait_sec"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    # Add derived metrics
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"] / 3600.0
    bottleneck_stats["cv"] = bottleneck_stats["std"] / bottleneck_stats["mean"]
    
    # Better handling of NaN values for coefficient of variation
    bottleneck_stats["cv"] = bottleneck_stats["cv"].fillna(0)
    
    # Calculate overall wait time percentiles
    wait_time_percentiles = np.percentile(transitions["wait_sec"], [50, 75, 90, 95, 99])
    percentile_labels = ["p50", "p75", "p90", "p95", "p99"]
    percentile_dict = dict(zip(percentile_labels, wait_time_percentiles))
    
    logger.debug(f"Wait time percentiles: {percentile_dict}")
    
    # Set the bottleneck threshold based on percentile
    bottleneck_threshold = np.percentile(transitions["wait_sec"], percentile_threshold)
    
    # Identify bottlenecks more efficiently
    # A bottleneck is a transition with wait time above threshold and sufficient frequency
    significant_bottlenecks = bottleneck_stats[
        (bottleneck_stats["mean"] > bottleneck_threshold) & 
        (bottleneck_stats["count"] >= freq_threshold)
    ].copy()
    
    # Add significance score
    significant_bottlenecks["bottleneck_score"] = (
        significant_bottlenecks["mean"] / bottleneck_threshold * 
        np.log1p(significant_bottlenecks["count"] / freq_threshold)
    )
    
    # Sort by bottleneck score
    significant_bottlenecks = significant_bottlenecks.sort_values(
        "bottleneck_score", ascending=False
    )
    
    # Add a rank column
    bottleneck_stats["rank"] = bottleneck_stats["mean"].rank(ascending=False)
    significant_bottlenecks["rank"] = significant_bottlenecks["bottleneck_score"].rank(ascending=False)
    
    # Log summary statistics
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    logger.info(
        f"Found {len(significant_bottlenecks)} significant bottlenecks "
        f"from {len(bottleneck_stats)} transitions"
    )
    
    # Add additional metadata
    bottleneck_stats.attrs["percentiles"] = percentile_dict
    bottleneck_stats.attrs["bottleneck_threshold"] = bottleneck_threshold
    bottleneck_stats.attrs["analysis_time"] = time.time() - start_time
    
    return bottleneck_stats, significant_bottlenecks

def analyze_cycle_times_vectorized(df: pd.DataFrame, 
                                  percentile_threshold: float = 95.0) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Analyze cycle times with optimized vector operations
    
    Args:
        df: Process data dataframe
        percentile_threshold: Percentile for identifying long-running cases
        
    Returns:
        Tuple of (case_stats, long_cases, percentile_value)
    """
    logger.info("Analyzing cycle times with vectorized operations")
    start_time = time.time()
    
    # Group by case and calculate min/max timestamps in a single operation
    case_stats = df.groupby("case_id")["timestamp"].agg(["min", "max"]).copy()
    
    # Calculate durations vectorized
    case_stats["duration"] = case_stats["max"] - case_stats["min"]
    case_stats["duration_h"] = case_stats["duration"].dt.total_seconds() / 3600.0
    case_stats["duration_days"] = case_stats["duration"].dt.total_seconds() / (3600.0 * 24)
    
    # Add case-level feature aggregations efficiently
    case_features = df.groupby("case_id").agg({
        "task_id": ["count", "nunique"],
        "resource_id": "nunique"
    })
    
    # Flatten multi-level columns for easier access
    case_features.columns = [f"{col[0]}_{col[1]}" for col in case_features.columns]
    
    # Merge features with case stats
    case_stats = case_stats.join(case_features)
    
    # Reset index for easier manipulation
    case_stats = case_stats.reset_index()
    
    # Calculate percentiles for duration
    percentiles = np.percentile(case_stats["duration_h"], [50, 75, 90, 95, 99])
    percentile_dict = {
        "p50": percentiles[0],
        "p75": percentiles[1],
        "p90": percentiles[2],
        "p95": percentiles[3],
        "p99": percentiles[4]
    }
    
    # Get the specified percentile value
    percentile_value = percentiles[3]  # p95 by default
    
    # Identify long-running cases
    long_cases = case_stats[case_stats["duration_h"] > percentile_value].copy()
    
    # Calculate correlation with features
    corr_events = case_stats["duration_h"].corr(case_stats["task_id_count"])
    corr_unique_tasks = case_stats["duration_h"].corr(case_stats["task_id_nunique"])
    corr_resources = case_stats["duration_h"].corr(case_stats["resource_id_nunique"])
    
    # Add correlation info to case_stats attributes for later use
    case_stats.attrs["duration_correlations"] = {
        "events": corr_events,
        "unique_tasks": corr_unique_tasks,
        "resources": corr_resources
    }
    
    # Add percentiles to attributes
    case_stats.attrs["percentiles"] = percentile_dict
    case_stats.attrs["analysis_time"] = time.time() - start_time
    
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    logger.info(
        f"Analyzed {len(case_stats)} cases, identified {len(long_cases)} "
        f"long-running cases (>{percentile_value:.1f}h)"
    )
    
    return case_stats, long_cases, percentile_value

def identify_process_variants(df: pd.DataFrame, 
                             max_variants: int = 10) -> Tuple[pd.DataFrame, Dict[int, List]]:
    """
    Identify process variants (unique sequences of activities)
    
    Args:
        df: Process data dataframe
        max_variants: Maximum number of variants to return details for
        
    Returns:
        Tuple of (variant_stats, variant_sequences)
    """
    logger.info("Identifying process variants")
    start_time = time.time()
    
    # Group by case and extract sequences of activities
    case_sequences = {}
    variant_sequences = {}
    variant_ids = {}
    
    # Pre-sort the dataframe by case and timestamp
    df_sorted = df.sort_values(["case_id", "timestamp"])
    
    # Get unique cases
    case_ids = df_sorted["case_id"].unique()
    
    # Process in batches for better memory usage
    batch_size = 1000
    
    for i in range(0, len(case_ids), batch_size):
        batch_case_ids = case_ids[i:i+batch_size]
        
        # Extract sequences for this batch
        for case_id, group in df_sorted[df_sorted["case_id"].isin(batch_case_ids)].groupby("case_id"):
            # Convert sequence to tuple for hashing
            sequence = tuple(group["task_id"].values)
            case_sequences[case_id] = sequence
            
            # Track unique variants
            if sequence in variant_ids:
                variant_ids[sequence] += 1
            else:
                variant_id = len(variant_ids) + 1
                variant_ids[sequence] = 1
                variant_sequences[variant_id] = list(sequence)
    
    # Create variant statistics
    variant_counts = pd.Series(variant_ids).reset_index()
    variant_counts.columns = ["sequence", "count"]
    variant_counts["percentage"] = variant_counts["count"] / len(case_ids) * 100
    variant_counts["variant_id"] = range(1, len(variant_counts) + 1)
    
    # Calculate sequence length
    variant_counts["sequence_length"] = variant_counts["sequence"].apply(len)
    
    # Sort by frequency
    variant_stats = variant_counts.sort_values("count", ascending=False).reset_index(drop=True)
    
    # Get top variants for detailed analysis
    top_variants = variant_stats.head(max_variants)
    top_variant_sequences = {
        row["variant_id"]: list(row["sequence"])
        for _, row in top_variants.iterrows()
    }
    
    logger.info(f"Identified {len(variant_stats)} unique process variants")
    logger.info(f"Top variant covers {variant_stats['percentage'].max():.1f}% of cases")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return variant_stats, top_variant_sequences

def analyze_resource_workload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze resource workload and efficiency
    
    Args:
        df: Process data dataframe
        
    Returns:
        DataFrame with resource workload statistics
    """
    logger.info("Analyzing resource workload")
    start_time = time.time()
    
    # Calculate activity counts per resource
    resource_stats = df.groupby("resource_id").agg({
        "task_id": "count",
        "case_id": "nunique"
    }).rename(columns={
        "task_id": "activity_count", 
        "case_id": "case_count"
    })
    
    # Calculate unique activities per resource
    resource_tasks = df.groupby("resource_id")["task_id"].nunique().rename("unique_activities")
    resource_stats = resource_stats.join(resource_tasks)
    
    # Calculate specialization ratio (higher = more specialized)
    resource_stats["specialization"] = 1 - (resource_stats["unique_activities"] / df["task_id"].nunique())
    
    # Calculate load distribution
    total_activities = df.shape[0]
    resource_stats["workload_percentage"] = resource_stats["activity_count"] / total_activities * 100
    
    # Sort by workload
    resource_stats = resource_stats.sort_values("activity_count", ascending=False)
    
    # Calculate workload distribution metrics
    gini = calculate_gini_coefficient(resource_stats["activity_count"].values)
    
    logger.info(f"Analyzed workload for {len(resource_stats)} resources")
    logger.info(f"Workload inequality (Gini): {gini:.3f}")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Store additional metrics in attributes
    resource_stats.attrs["gini_coefficient"] = gini
    resource_stats.attrs["analysis_time"] = time.time() - start_time
    
    return resource_stats

def calculate_gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for measuring inequality
    
    Args:
        values: Array of values (e.g., workload per resource)
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = complete inequality)
    """
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    if n == 0 or np.sum(sorted_values) == 0:
        return 0
    
    # Calculate cumulative sum of sorted array
    cumsum = np.cumsum(sorted_values)
    
    # Gini = 1 - 2 * sum of the area under the Lorenz curve
    return (n + 1 - 2 * np.sum(cumsum) / np.sum(sorted_values)) / n