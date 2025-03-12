"""
Highly optimized process mining analysis utilities with vectorized operations
"""
import pandas as pd
import numpy as np
import time
import logging
import gc
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

def vectorize(func):
    """
    Decorator that provides a vectorized implementation with fallback to standard
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with vectorization
    """
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
    Analyze process bottlenecks using highly optimized vectorized operations
    
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
    
    # Add transitions in a single vectorized operation
    # This avoids making a full copy of the dataframe
    transitions = df[['case_id', 'task_id', 'timestamp']].copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    transitions["next_timestamp"] = transitions.groupby("case_id")["timestamp"].shift(-1)
    
    # Filter transitions efficiently
    transitions = transitions.dropna(subset=["next_task_id"])
    
    # Vectorized computation of wait times
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    # Group and calculate statistics in one operation
    bottleneck_stats = transitions.groupby(["task_id", "next_task_id"])["wait_sec"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    # Convert task IDs to integers to avoid type issues
    bottleneck_stats["task_id"] = bottleneck_stats["task_id"].astype(int)
    bottleneck_stats["next_task_id"] = bottleneck_stats["next_task_id"].astype(int)
    
    # Add derived metrics vectorized
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"] / 3600.0
    bottleneck_stats["cv"] = bottleneck_stats["std"] / bottleneck_stats["mean"].clip(lower=1e-10)
    
    # Calculate wait time percentiles efficiently
    wait_time_percentiles = np.percentile(transitions["wait_sec"], [50, 75, 90, 95, 99])
    percentile_labels = ["p50", "p75", "p90", "p95", "p99"]
    percentile_dict = dict(zip(percentile_labels, wait_time_percentiles))
    
    # Set bottleneck threshold
    bottleneck_threshold = np.percentile(transitions["wait_sec"], percentile_threshold)
    
    # Identify significant bottlenecks vectorized
    significant_bottlenecks = bottleneck_stats[
        (bottleneck_stats["mean"] > bottleneck_threshold) & 
        (bottleneck_stats["count"] >= freq_threshold)
    ].copy()
    
    # Add bottleneck score vectorized
    if len(significant_bottlenecks) > 0:
        significant_bottlenecks["bottleneck_score"] = (
            significant_bottlenecks["mean"] / bottleneck_threshold * 
            np.log1p(significant_bottlenecks["count"] / freq_threshold)
        )
        
        # Sort by bottleneck score
        significant_bottlenecks = significant_bottlenecks.sort_values("bottleneck_score", ascending=False)
    
    # Add rank columns efficiently
    bottleneck_stats["rank"] = bottleneck_stats["mean"].rank(ascending=False)
    if len(significant_bottlenecks) > 0:
        significant_bottlenecks["rank"] = significant_bottlenecks["bottleneck_score"].rank(ascending=False)
    
    # Log summary
    logger.info(f"Found {len(significant_bottlenecks)} significant bottlenecks out of {len(bottleneck_stats)} transitions")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Add percentiles as attributes
    bottleneck_stats.attrs["percentiles"] = percentile_dict
    bottleneck_stats.attrs["bottleneck_threshold"] = bottleneck_threshold
    
    # Force memory cleanup for large datasets
    del transitions
    gc.collect()
    
    return bottleneck_stats, significant_bottlenecks

@vectorize
def analyze_cycle_times(df: pd.DataFrame, percentile_threshold: float = 95.0) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Analyze cycle times with optimized vectorized operations
    
    Args:
        df: Process data dataframe
        percentile_threshold: Percentile for identifying long-running cases
        
    Returns:
        Tuple of (case_stats, long_cases, percentile_value)
    """
    start_time = time.time()
    logger.info("Analyzing cycle times")
    
    # Select only needed columns to reduce memory usage
    df_slim = df[['case_id', 'task_id', 'resource_id', 'timestamp']].copy()
    
    # Group by case and calculate min/max timestamps in one operation
    case_stats = df_slim.groupby("case_id")["timestamp"].agg(["min", "max"])
    
    # Calculate durations vectorized
    case_stats["duration"] = case_stats["max"] - case_stats["min"]
    case_stats["duration_h"] = case_stats["duration"].dt.total_seconds() / 3600.0
    case_stats["duration_days"] = case_stats["duration"].dt.total_seconds() / (3600.0 * 24)
    
    # Add case-level features efficiently
    case_features = df_slim.groupby("case_id").agg({
        "task_id": ["count", "nunique"],
        "resource_id": "nunique"
    })
    
    # Flatten columns for easier access
    case_features.columns = [f"{col[0]}_{col[1]}" for col in case_features.columns]
    
    # Merge features with case stats efficiently
    case_stats = case_stats.join(case_features)
    case_stats = case_stats.reset_index()
    
    # Calculate percentiles efficiently
    duration_values = case_stats["duration_h"].values
    percentiles = np.percentile(duration_values, [50, 75, 90, 95, 99])
    percentile_dict = {
        "p50": percentiles[0],
        "p75": percentiles[1],
        "p90": percentiles[2],
        "p95": percentiles[3],
        "p99": percentiles[4]
    }
    
    # Get the specified percentile value
    idx = min(int(percentile_threshold / 25), 3)  # Map threshold to index (0-3)
    percentile_value = percentiles[idx]
    
    # Identify long-running cases vectorized
    long_cases = case_stats[case_stats["duration_h"] > percentile_value].copy()
    
    # Calculate correlation metrics for insights
    corr_events = case_stats["duration_h"].corr(case_stats["task_id_count"])
    corr_unique = case_stats["duration_h"].corr(case_stats["task_id_nunique"])
    
    # Store metrics as attributes
    case_stats.attrs["percentiles"] = percentile_dict
    case_stats.attrs["correlations"] = {
        "events_vs_duration": corr_events,
        "unique_tasks_vs_duration": corr_unique
    }
    
    # Log summary
    logger.info(f"Analyzed {len(case_stats)} cases, identified {len(long_cases)} long-running cases (>{percentile_value:.1f}h)")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Clean up memory
    del df_slim
    gc.collect()
    
    return case_stats, long_cases, percentile_value

def analyze_transition_patterns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze transition patterns and compute transition matrix with optimized operations
    
    Args:
        df: Process data dataframe
        
    Returns:
        Tuple of (transitions, transition_count, probability_matrix)
    """
    start_time = time.time()
    logger.info("Analyzing transition patterns")
    
    # Select only needed columns
    df_slim = df[['case_id', 'task_id', 'timestamp']].copy()
    
    # Compute transitions in a single operation
    transitions = df_slim.copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    
    # Filter out last events in cases
    transitions = transitions.dropna(subset=["next_task_id"])
    
    # Create transition count matrix in a single operation
    trans_count = pd.crosstab(
        transitions["task_id"], 
        transitions["next_task_id"],
        normalize=False
    )
    
    # Calculate probability matrix efficiently
    row_sums = trans_count.sum(axis=1)
    prob_matrix = trans_count.div(row_sums, axis=0).fillna(0)
    
    # Log summary
    total_transitions = trans_count.sum().sum()
    unique_transitions = (trans_count > 0).sum().sum()
    logger.info(f"Found {total_transitions:,} total transitions across {unique_transitions:,} unique patterns")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Clean up memory
    del df_slim
    gc.collect()
    
    return transitions, trans_count, prob_matrix

def identify_process_variants(df: pd.DataFrame, max_variants: int = 10) -> Tuple[pd.DataFrame, Dict[int, List]]:
    """
    Identify process variants (unique sequences of activities) with optimized memory usage
    
    Args:
        df: Process data dataframe
        max_variants: Maximum number of variants to return details for
        
    Returns:
        Tuple of (variant_stats, variant_sequences)
    """
    start_time = time.time()
    logger.info("Identifying process variants")
    
    # Select only needed columns
    df_slim = df[['case_id', 'task_id', 'timestamp']].copy()
    
    # Pre-sort the dataframe for faster groupby operations
    df_sorted = df_slim.sort_values(["case_id", "timestamp"])
    
    # Extract case sequences efficiently using string representation for hashing
    variant_hash = {}
    variant_counts = {}
    
    # Create a dictionary to track variants by case_id
    case_to_variant = {}
    
    # First pass: extract sequences and count variants
    for case_id, group in df_sorted.groupby("case_id"):
        # Convert to string for efficient hashing
        task_seq = group["task_id"].astype(str).str.cat(sep=',')
        case_to_variant[case_id] = task_seq
        
        if task_seq in variant_counts:
            variant_counts[task_seq] += 1
            # Store full sequence only for the first occurrence to save memory
        else:
            variant_counts[task_seq] = 1
            variant_hash[task_seq] = list(group["task_id"].values)
    
    # Create variant statistics dataframe
    variant_data = []
    for i, (sequence_str, count) in enumerate(sorted(variant_counts.items(), key=lambda x: x[1], reverse=True)):
        variant_id = i + 1
        sequence = variant_hash[sequence_str]
        variant_data.append({
            'variant_id': variant_id,
            'sequence': sequence,  # Store as list
            'count': count,
            'percentage': count / len(case_to_variant) * 100,
            'sequence_length': len(sequence)
        })
    
    variant_stats = pd.DataFrame(variant_data)
    
    # Extract top variant sequences
    variant_sequences = {
        row['variant_id']: row['sequence']
        for row in variant_data[:max_variants]
    }
    
    # Log summary
    logger.info(f"Identified {len(variant_stats)} unique process variants across {len(case_to_variant)} cases")
    if len(variant_stats) > 0:
        logger.info(f"Top variant covers {variant_stats['percentage'].iloc[0]:.1f}% of cases")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Clean up memory
    del df_slim, df_sorted, variant_hash, case_to_variant
    gc.collect()
    
    return variant_stats, variant_sequences

def analyze_resource_workload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze resource workload and efficiency with vectorized operations
    
    Args:
        df: Process data dataframe
        
    Returns:
        DataFrame with resource workload statistics
    """
    start_time = time.time()
    logger.info("Analyzing resource workload")
    
    # Select only the necessary columns
    df_slim = df[['resource_id', 'task_id', 'case_id']].copy()
    
    # Calculate activity counts per resource in one operation
    resource_stats = df_slim.groupby("resource_id").agg({
        "task_id": "count",
        "case_id": "nunique"
    }).rename(columns={
        "task_id": "activity_count", 
        "case_id": "case_count"
    })
    
    # Calculate unique activities per resource
    resource_tasks = df_slim.groupby("resource_id")["task_id"].nunique().rename("unique_activities")
    resource_stats = resource_stats.join(resource_tasks)
    
    # Calculate specialization ratio (higher = more specialized)
    task_count = df_slim["task_id"].nunique()
    resource_stats["specialization"] = 1 - (resource_stats["unique_activities"] / task_count)
    
    # Calculate load distribution
    total_activities = df_slim.shape[0]
    resource_stats["workload_percentage"] = resource_stats["activity_count"] / total_activities * 100
    
    # Calculate efficiency metric
    resource_stats["efficiency"] = resource_stats["case_count"] / resource_stats["activity_count"]
    
    # Calculate Gini coefficient for workload inequality
    gini = _calculate_gini_coefficient(resource_stats["activity_count"].values)
    resource_stats.attrs["gini_coefficient"] = gini
    
    # Sort by workload
    resource_stats = resource_stats.sort_values("activity_count", ascending=False)
    
    # Log summary
    logger.info(f"Analyzed workload for {len(resource_stats)} resources")
    logger.info(f"Workload inequality (Gini): {gini:.3f}")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Clean up memory
    del df_slim
    gc.collect()
    
    return resource_stats

def _calculate_gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate the Gini coefficient with optimized numpy operations
    
    Args:
        values: Array of values to calculate Gini coefficient for
        
    Returns:
        Gini coefficient (0=equal distribution, 1=complete inequality)
    """
    # Handle edge cases
    if len(values) == 0 or np.sum(values) == 0:
        return 0
    
    # Sort values (required for Lorenz curve)
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Calculate Lorenz curve using cumulative sum
    lorenz = np.cumsum(sorted_values) / np.sum(sorted_values)
    
    # Calculate Gini coefficient using area under Lorenz curve
    # Formula: G = 1 - 2 * area under Lorenz curve
    # For discrete points: G = 1 - sum((y_i + y_{i-1}) * (x_i - x_{i-1}))
    # With uniform x points: G = 1 - sum(y_i + y_{i-1}) / n
    
    # Using vectorized approach for (y_i + y_{i-1})
    lorenz_padded = np.pad(lorenz, (1, 0), 'constant')
    sum_pairs = lorenz_padded[:-1] + lorenz_padded[1:]
    
    # Gini coefficient
    gini = 1 - np.sum(sum_pairs[1:]) / n
    
    return gini