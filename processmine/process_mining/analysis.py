import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

def vectorize(func):
    """Decorator for vectorized versions of analysis functions"""
    def wrapper(*args, **kwargs):
        # Check if vectorized version is enabled
        use_vectorized = kwargs.pop('vectorized', True)
        
        if use_vectorized:
            try:
                # Try to use vectorized implementation
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Vectorized implementation failed: {e}. Falling back to standard.")
                return func(*args, vectorized=False, **kwargs)
        else:
            return func(*args, **kwargs)
            
    return wrapper

@vectorize
def analyze_bottlenecks(df: pd.DataFrame, 
                       freq_threshold: int = 5,
                       percentile_threshold: float = 90.0,
                       vectorized: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze process bottlenecks using vectorized operations
    
    Args:
        df: Process data dataframe
        freq_threshold: Minimum frequency threshold for significant bottlenecks
        percentile_threshold: Percentile threshold for identifying bottlenecks
        vectorized: Whether to use vectorized implementation
        
    Returns:
        Tuple of (bottleneck_stats, significant_bottlenecks)
    """
    start_time = time.time()
    logger.info("Analyzing process bottlenecks")
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Add next_task_id and next_timestamp columns
    df_copy["next_task_id"] = df_copy.groupby("case_id")["task_id"].shift(-1)
    df_copy["next_timestamp"] = df_copy.groupby("case_id")["timestamp"].shift(-1)
    
    # Filter transitions
    transitions = df_copy.dropna(subset=["next_task_id"]).copy()
    
    # Compute wait times
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    # Group and calculate statistics
    bottleneck_stats = transitions.groupby(["task_id", "next_task_id"])["wait_sec"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    # Add derived metrics
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"] / 3600.0
    bottleneck_stats["cv"] = bottleneck_stats["std"] / bottleneck_stats["mean"].clip(lower=1e-10)
    
    # Calculate wait time percentiles
    wait_time_percentiles = np.percentile(transitions["wait_sec"], [50, 75, 90, 95, 99])
    percentile_labels = ["p50", "p75", "p90", "p95", "p99"]
    percentile_dict = dict(zip(percentile_labels, wait_time_percentiles))
    
    # Set bottleneck threshold
    bottleneck_threshold = np.percentile(transitions["wait_sec"], percentile_threshold)
    
    # Identify significant bottlenecks
    significant_bottlenecks = bottleneck_stats[
        (bottleneck_stats["mean"] > bottleneck_threshold) & 
        (bottleneck_stats["count"] >= freq_threshold)
    ].copy()
    
    # Add bottleneck score
    significant_bottlenecks["bottleneck_score"] = (
        significant_bottlenecks["mean"] / bottleneck_threshold * 
        np.log1p(significant_bottlenecks["count"] / freq_threshold)
    )
    
    # Sort by bottleneck score
    significant_bottlenecks = significant_bottlenecks.sort_values("bottleneck_score", ascending=False)
    
    # Add rank columns
    bottleneck_stats["rank"] = bottleneck_stats["mean"].rank(ascending=False)
    significant_bottlenecks["rank"] = significant_bottlenecks["bottleneck_score"].rank(ascending=False)
    
    # Log summary
    logger.info(f"Found {len(significant_bottlenecks)} significant bottlenecks out of {len(bottleneck_stats)} transitions")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Add percentiles as attributes
    bottleneck_stats.attrs["percentiles"] = percentile_dict
    bottleneck_stats.attrs["bottleneck_threshold"] = bottleneck_threshold
    
    return bottleneck_stats, significant_bottlenecks

@vectorize
def analyze_cycle_times(df: pd.DataFrame, percentile_threshold: float = 95.0) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Analyze cycle times with vectorized operations
    
    Args:
        df: Process data dataframe
        percentile_threshold: Percentile for identifying long-running cases
        
    Returns:
        Tuple of (case_stats, long_cases, percentile_value)
    """
    start_time = time.time()
    logger.info("Analyzing cycle times")
    
    # Group by case and calculate min/max timestamps
    case_stats = df.groupby("case_id")["timestamp"].agg(["min", "max"]).copy()
    
    # Calculate durations
    case_stats["duration"] = case_stats["max"] - case_stats["min"]
    case_stats["duration_h"] = case_stats["duration"].dt.total_seconds() / 3600.0
    case_stats["duration_days"] = case_stats["duration"].dt.total_seconds() / (3600.0 * 24)
    
    # Add case-level features
    case_features = df.groupby("case_id").agg({
        "task_id": ["count", "nunique"],
        "resource_id": "nunique"
    })
    
    # Flatten columns for easier access
    case_features.columns = [f"{col[0]}_{col[1]}" for col in case_features.columns]
    
    # Merge features with case stats
    case_stats = case_stats.join(case_features)
    case_stats = case_stats.reset_index()
    
    # Calculate percentiles
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
    
    # Store percentiles as attributes
    case_stats.attrs["percentiles"] = percentile_dict
    
    # Log summary
    logger.info(f"Analyzed {len(case_stats)} cases, identified {len(long_cases)} long-running cases (>{percentile_value:.1f}h)")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return case_stats, long_cases, percentile_value

def analyze_transition_patterns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze transition patterns and compute transition matrix
    
    Args:
        df: Process data dataframe
        
    Returns:
        Tuple of (transitions, transition_count, probability_matrix)
    """
    start_time = time.time()
    logger.info("Analyzing transition patterns")
    
    # Compute transitions
    transitions = df.copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    
    # Filter out last events in cases
    transitions = transitions.dropna(subset=["next_task_id"])
    
    # Create transition count matrix and probability matrix
    trans_count = transitions.groupby(["task_id", "next_task_id"]).size().unstack(fill_value=0)
    prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
    
    # Log summary
    total_transitions = trans_count.sum().sum()
    unique_transitions = (trans_count > 0).sum().sum()
    logger.info(f"Found {total_transitions} total transitions with {unique_transitions} unique patterns")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return transitions, trans_count, prob_matrix

def identify_process_variants(df: pd.DataFrame, max_variants: int = 10) -> Tuple[pd.DataFrame, Dict[int, List]]:
    """
    Identify process variants (unique sequences of activities)
    
    Args:
        df: Process data dataframe
        max_variants: Maximum number of variants to return details for
        
    Returns:
        Tuple of (variant_stats, variant_sequences)
    """
    start_time = time.time()
    logger.info("Identifying process variants")
    
    # Pre-sort the dataframe
    df_sorted = df.sort_values(["case_id", "timestamp"])
    
    # Extract case sequences efficiently
    case_sequences = {}
    variant_counts = {}
    
    # Process each case
    for case_id, group in df_sorted.groupby("case_id"):
        # Convert to tuple for hashing
        sequence = tuple(group["task_id"].values)
        case_sequences[case_id] = sequence
        
        # Count variants
        if sequence in variant_counts:
            variant_counts[sequence] += 1
        else:
            variant_counts[sequence] = 1
    
    # Create variant statistics
    variant_data = []
    for i, (sequence, count) in enumerate(sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)):
        variant_data.append({
            'variant_id': i+1,
            'sequence': sequence,
            'count': count,
            'percentage': count / len(case_sequences) * 100,
            'sequence_length': len(sequence)
        })
    
    variant_stats = pd.DataFrame(variant_data)
    
    # Extract top variant sequences
    variant_sequences = {
        row['variant_id']: list(row['sequence'])
        for row in variant_data[:max_variants]
    }
    
    # Log summary
    logger.info(f"Identified {len(variant_stats)} unique process variants")
    logger.info(f"Top variant covers {variant_stats['percentage'].max():.1f}% of cases")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return variant_stats, variant_sequences

def analyze_resource_workload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze resource workload and efficiency
    
    Args:
        df: Process data dataframe
        
    Returns:
        DataFrame with resource workload statistics
    """
    start_time = time.time()
    logger.info("Analyzing resource workload")
    
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
    
    # Calculate Gini coefficient for workload inequality
    gini = _calculate_gini_coefficient(resource_stats["activity_count"].values)
    resource_stats.attrs["gini_coefficient"] = gini
    
    # Sort by workload
    resource_stats = resource_stats.sort_values("activity_count", ascending=False)
    
    # Log summary
    logger.info(f"Analyzed workload for {len(resource_stats)} resources")
    logger.info(f"Workload inequality (Gini): {gini:.3f}")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return resource_stats

def _calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate the Gini coefficient for measuring inequality"""
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    if n == 0 or np.sum(sorted_values) == 0:
        return 0
    
    # Calculate cumulative sum of sorted array
    cumsum = np.cumsum(sorted_values)
    
    # Gini = 1 - 2 * sum of the area under the Lorenz curve
    return (n + 1 - 2 * np.sum(cumsum) / np.sum(sorted_values)) / n

def perform_conformance_checking(df: pd.DataFrame) -> Tuple[List, int]:
    """
    Simplified conformance checking without PM4Py dependency
    
    Args:
        df: Process data dataframe
        
    Returns:
        Tuple of (replayed_traces, num_deviant_traces)
    """
    start_time = time.time()
    logger.info("Performing conformance checking")
    
    # Extract variants
    variant_stats, _ = identify_process_variants(df)
    
    # Use top variant as reference model (simplified approach)
    top_variant_sequence = eval(variant_stats.iloc[0]['sequence'])
    top_variant_count = variant_stats.iloc[0]['count']
    total_variants = variant_stats['count'].sum()
    
    # Count non-conforming traces (those not following top variant)
    n_deviant = total_variants - top_variant_count
    
    # Create a simple representation of replayed traces
    replayed = [{'is_fit': True}] * top_variant_count + [{'is_fit': False}] * n_deviant
    
    conformance = top_variant_count / total_variants
    
    # Log summary
    logger.info(f"Conformance check: {conformance:.1%} traces conform to the top variant")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return replayed, n_deviant

# Functions to incorporate from analysis.py
def build_task_adjacency(df: pd.DataFrame, num_tasks: int) -> np.ndarray:
    """
    Build adjacency matrix weighted by transition frequencies
    
    Args:
        df: Process data dataframe
        num_tasks: Number of tasks
        
    Returns:
        Weighted adjacency matrix
    """
    # Implementation from analysis.py
    start_time = time.time()
    logger.info("Building task adjacency matrix")
    
    # Initialize matrix
    A = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    
    # Process transitions
    for cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        tasks_seq = cdata["task_id"].values
        for i in range(len(tasks_seq)-1):
            src = int(tasks_seq[i])
            tgt = int(tasks_seq[i+1])
            A[src, tgt] += 1.0
    
    # Add reverse edges for undirected clustering
    A_sym = A + A.T
    
    # Log summary
    non_zero = np.count_nonzero(A)
    density = non_zero / (num_tasks * num_tasks)
    logger.info(f"Built adjacency matrix {A.shape} with {non_zero} non-zero entries ({density:.1%} density)")
    logger.info(f"Matrix built in {time.time() - start_time:.2f}s")
    
    return A_sym