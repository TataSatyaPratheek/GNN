"""
Fully vectorized process mining analysis with minimal memory usage
and optimized algorithms for large-scale event logs.
"""
import pandas as pd
import numpy as np
import time
import logging
import gc
from typing import Dict, List, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

def analyze_bottlenecks(
    df: pd.DataFrame, 
    freq_threshold: int = 5,
    percentile_threshold: float = 90.0,
    vectorized: bool = True,
    memory_efficient: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze process bottlenecks with fully vectorized operations and minimal memory usage
    
    Args:
        df: Process data dataframe
        freq_threshold: Minimum frequency threshold for significant bottlenecks
        percentile_threshold: Percentile threshold for identifying bottlenecks
        vectorized: Whether to use vectorized implementation (faster but uses more memory)
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        Tuple of (bottleneck_stats, significant_bottlenecks)
    """
    start_time = time.time()
    logger.info("Analyzing process bottlenecks...")
    
    # Create transitions dataframe - this is the core computation
    # Only select necessary columns to reduce memory usage
    if memory_efficient:
        # Memory efficient approach creates smaller intermediate dataframes
        transitions = _create_transitions_efficient(df, ['case_id', 'task_id', 'timestamp'])
    else:
        # Standard approach that may use more memory but is simpler
        transitions = _create_transitions_standard(df)
    
    # Compute wait times in seconds
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    # Free original dataframe memory if processing large datasets
    if memory_efficient and len(df) > 1000000:
        # We only need transitions now, allow original dataframe to be garbage collected
        tmp_df = df
        df = None
        del tmp_df
        gc.collect()
    
    # Vectorized calculation of statistics
    bottleneck_stats = _calculate_bottleneck_stats(transitions)
    
    # Calculate wait time percentiles for threshold
    wait_time_percentiles = np.percentile(transitions["wait_sec"], [50, 75, 90, 95, 99])
    percentile_dict = dict(zip(["p50", "p75", "p90", "p95", "p99"], wait_time_percentiles))
    
    # Set bottleneck threshold based on percentile
    bottleneck_threshold = np.percentile(transitions["wait_sec"], percentile_threshold)
    
    # Identify significant bottlenecks
    significant_bottlenecks = _identify_significant_bottlenecks(
        bottleneck_stats, bottleneck_threshold, freq_threshold
    )
    
    # Add rank columns
    bottleneck_stats["rank"] = bottleneck_stats["mean"].rank(ascending=False)
    if len(significant_bottlenecks) > 0:
        significant_bottlenecks["rank"] = significant_bottlenecks["bottleneck_score"].rank(ascending=False)
    
    # Store additional metadata
    bottleneck_stats.attrs["percentiles"] = percentile_dict
    bottleneck_stats.attrs["bottleneck_threshold"] = bottleneck_threshold
    bottleneck_stats.attrs["analysis_time"] = time.time() - start_time
    
    # Log summary
    logger.info(f"Found {len(significant_bottlenecks)} significant bottlenecks out of {len(bottleneck_stats)} transitions")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Free memory
    if memory_efficient:
        del transitions
        gc.collect()
    
    return bottleneck_stats, significant_bottlenecks

def _create_transitions_efficient(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Create transitions dataframe with minimal memory usage
    
    Args:
        df: Original dataframe
        columns: Columns to include in transitions
        
    Returns:
        Transitions dataframe
    """
    # Create minimal transitions dataframe with only necessary columns
    transitions = df[columns].copy()
    
    # Add next task and timestamp using shift operation
    # This will significantly reduce memory usage for large dataframes
    transitions = transitions.sort_values(['case_id', 'timestamp'])
    
    # Group by case_id and apply shift for each group
    grouped = transitions.groupby('case_id')
    
    # Use faster approach with vectorized shift
    transitions['next_task_id'] = grouped['task_id'].shift(-1).values
    transitions['next_timestamp'] = grouped['timestamp'].shift(-1).values
    
    # Filter transitions efficiently - remove last event in each case
    transitions = transitions.dropna(subset=['next_task_id'])
    
    # Ensure proper data types to save memory
    transitions = transitions.copy()  # Create explicit copy first
    transitions['next_task_id'] = transitions['next_task_id'].astype(np.int32)
    
    return transitions

def _create_transitions_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transitions dataframe - standard approach 
    """
    # Create full copy of needed columns
    transitions = df[['case_id', 'task_id', 'timestamp']].copy()
    
    # Add next task and timestamp
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    transitions["next_timestamp"] = transitions.groupby("case_id")["timestamp"].shift(-1)
    
    # Filter transitions
    transitions = transitions.dropna(subset=["next_task_id"])
    transitions["next_task_id"] = transitions["next_task_id"].astype(int)
    
    return transitions

def _calculate_bottleneck_stats(transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate bottleneck statistics using vectorized operations
    
    Args:
        transitions: Transitions dataframe
        
    Returns:
        Dataframe with bottleneck statistics
    """
    # Group and calculate statistics in one vectorized operation
    # This avoids loops and is much faster
    bottleneck_stats = transitions.groupby(["task_id", "next_task_id"])["wait_sec"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    # Handle potential NaN values in std calculation
    bottleneck_stats["std"] = bottleneck_stats["std"].fillna(0)
    
    # Add derived metrics vectorized
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"] / 3600.0
    
    # Calculate coefficient of variation properly handling zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        bottleneck_stats["cv"] = bottleneck_stats["std"] / bottleneck_stats["mean"]
    
    bottleneck_stats["cv"] = bottleneck_stats["cv"].fillna(0).replace([np.inf, -np.inf], 0)
    
    return bottleneck_stats

def _identify_significant_bottlenecks(
    bottleneck_stats: pd.DataFrame, 
    bottleneck_threshold: float, 
    freq_threshold: int
) -> pd.DataFrame:
    """
    Identify significant bottlenecks with vectorized filtering
    
    Args:
        bottleneck_stats: Bottleneck statistics dataframe
        bottleneck_threshold: Threshold for bottleneck mean wait time
        freq_threshold: Threshold for bottleneck frequency
        
    Returns:
        Dataframe with significant bottlenecks
    """
    # Vectorized filtering for bottlenecks (faster than iterating)
    significant_bottlenecks = bottleneck_stats[
        (bottleneck_stats["mean"] > bottleneck_threshold) & 
        (bottleneck_stats["count"] >= freq_threshold)
    ].copy()
    
    if len(significant_bottlenecks) > 0:
        # Calculate bottleneck score in a single vectorized operation
        # Higher score means more significant bottleneck
        significant_bottlenecks["bottleneck_score"] = (
            significant_bottlenecks["mean"] / bottleneck_threshold * 
            np.log1p(significant_bottlenecks["count"] / freq_threshold)
        )
        
        # Sort by bottleneck score
        significant_bottlenecks = significant_bottlenecks.sort_values(
            "bottleneck_score", ascending=False
        )
    
    return significant_bottlenecks

def analyze_cycle_times(
    df: pd.DataFrame, 
    percentile_threshold: float = 95.0,
    memory_efficient: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Analyze case cycle times with vectorized operations and memory optimization
    
    Args:
        df: Process data dataframe
        percentile_threshold: Percentile for identifying long-running cases
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        Tuple of (case_stats, long_cases, percentile_value)
    """
    start_time = time.time()
    logger.info("Analyzing cycle times...")
    
    if memory_efficient:
        # Only keep necessary columns for this analysis
        df_slim = df[['case_id', 'task_id', 'resource_id', 'timestamp']].copy()
    else:
        df_slim = df
    
    # Group by case and calculate min/max timestamps in one operation
    # This is more efficient than looping through cases
    case_stats = df_slim.groupby("case_id")["timestamp"].agg(["min", "max"])
    
    # Calculate durations vectorized
    case_stats["duration"] = case_stats["max"] - case_stats["min"]
    case_stats["duration_h"] = case_stats["duration"].dt.total_seconds() / 3600.0
    case_stats["duration_days"] = case_stats["duration"].dt.total_seconds() / (3600.0 * 24)
    
    # Calculate additional case-level metrics in one grouped operation
    case_features = df_slim.groupby("case_id").agg({
        "task_id": ["count", "nunique"],
        "resource_id": "nunique"
    })
    
    # Flatten multi-level columns
    case_features.columns = [f"{col[0]}_{col[1]}" for col in case_features.columns]
    
    # Merge features with case stats using efficient join
    case_stats = case_stats.join(case_features)
    
    # Free memory
    if memory_efficient:
        del df_slim
        gc.collect()
    
    # Reset index for easier access
    case_stats = case_stats.reset_index()
    
    # Calculate percentiles efficiently using numpy
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
    idx = min(int(percentile_threshold / 25), 3)  # Map threshold to index
    percentile_value = percentiles[idx]
    
    # Identify long-running cases with vectorized filtering
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
    logger.info(f"Analyzed {len(case_stats)} cases, identified {len(long_cases)} " +
                f"long-running cases (>{percentile_value:.1f}h)")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return case_stats, long_cases, percentile_value

def analyze_transition_patterns(
    df: pd.DataFrame, 
    memory_efficient: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze transition patterns and compute transition matrix with optimized memory usage
    
    Args:
        df: Process data dataframe
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        Tuple of (transitions, transition_count, probability_matrix)
    """
    start_time = time.time()
    logger.info("Analyzing transition patterns...")
    
    if memory_efficient:
        # Select only needed columns
        df_slim = df[['case_id', 'task_id', 'timestamp']].copy()
    else:
        df_slim = df
    
    # Compute transitions efficiently
    transitions = df_slim.copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    
    # Filter out last events in cases
    transitions = transitions.dropna(subset=["next_task_id"])
    transitions["next_task_id"] = transitions["next_task_id"].astype(int)
    
    # Create transition count matrix using crosstab (highly optimized)
    trans_count = pd.crosstab(
        transitions["task_id"], 
        transitions["next_task_id"],
        normalize=False
    )
    
    # Calculate probability matrix with vectorized operations
    # Divide each row by its sum to get probabilities
    row_sums = trans_count.sum(axis=1)
    prob_matrix = trans_count.div(row_sums, axis=0).fillna(0)
    
    # Log summary
    total_transitions = trans_count.sum().sum()
    unique_transitions = (trans_count > 0).sum().sum()
    logger.info(f"Found {total_transitions:,} total transitions across {unique_transitions:,} unique patterns")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Free memory
    if memory_efficient:
        del df_slim
        gc.collect()
    
    return transitions, trans_count, prob_matrix

def identify_process_variants(
    df: pd.DataFrame, 
    max_variants: int = 10,
    memory_efficient: bool = True
) -> Tuple[pd.DataFrame, Dict[int, List]]:
    """
    Identify process variants (unique sequences of activities) with memory optimization
    
    Args:
        df: Process data dataframe
        max_variants: Maximum number of variants to return details for
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        Tuple of (variant_stats, variant_sequences)
    """
    start_time = time.time()
    logger.info("Identifying process variants...")
    
    if memory_efficient:
        # Pre-sort the dataframe for faster groupby operations
        df_sorted = df[['case_id', 'task_id', 'timestamp']].sort_values(["case_id", "timestamp"])
    else:
        df_sorted = df.sort_values(["case_id", "timestamp"])
    
    # Use dictionary for efficient variant counting
    variant_hash = {}
    variant_counts = {}
    
    # Create a dictionary to track variants by case_id
    case_to_variant = {}
    
    # Process in batches for very large datasets
    if memory_efficient and len(df_sorted) > 1000000:
        batch_size = 100000
        for start_idx in range(0, len(df_sorted), batch_size):
            end_idx = min(start_idx + batch_size, len(df_sorted))
            batch_df = df_sorted.iloc[start_idx:end_idx]
            
            # Process each case in the batch
            _process_variants_batch(batch_df, variant_hash, variant_counts, case_to_variant)
            
            # Force garbage collection
            del batch_df
            gc.collect()
    else:
        # Process in one go for smaller datasets
        _process_variants_batch(df_sorted, variant_hash, variant_counts, case_to_variant)
    
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
        for row in variant_data[:min(max_variants, len(variant_data))]
    }
    
    # Log summary
    logger.info(f"Identified {len(variant_stats)} unique process variants across {len(case_to_variant)} cases")
    if len(variant_stats) > 0:
        logger.info(f"Top variant covers {variant_stats['percentage'].iloc[0]:.1f}% of cases")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Free memory
    if memory_efficient:
        del df_sorted, variant_hash, case_to_variant
        gc.collect()
    
    return variant_stats, variant_sequences

def _process_variants_batch(df_batch, variant_hash, variant_counts, case_to_variant):
    """
    Process a batch of data to identify variants
    
    Args:
        df_batch: Batch of dataframe
        variant_hash: Dictionary to store variant sequences
        variant_counts: Dictionary to count variants
        case_to_variant: Dictionary to track variants by case
    """
    # Group by case_id and extract sequences
    for case_id, group in df_batch.groupby("case_id"):
        # Skip if already processed (for overlapping batches)
        if case_id in case_to_variant:
            continue
            
        # Convert to fast string representation for hashing
        task_seq = group["task_id"].astype(str).str.cat(sep=',')
        case_to_variant[case_id] = task_seq
        
        # Count variants
        if task_seq in variant_counts:
            variant_counts[task_seq] += 1
        else:
            variant_counts[task_seq] = 1
            variant_hash[task_seq] = list(group["task_id"].values)

def analyze_resource_workload(
    df: pd.DataFrame, 
    memory_efficient: bool = True
) -> pd.DataFrame:
    """
    Analyze resource workload and efficiency with vectorized operations
    
    Args:
        df: Process data dataframe
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        DataFrame with resource workload statistics
    """
    start_time = time.time()
    logger.info("Analyzing resource workload...")
    
    if memory_efficient:
        # Select only necessary columns
        df_slim = df[['resource_id', 'task_id', 'case_id']].copy()
    else:
        df_slim = df
    
    # Calculate activity counts per resource in optimized grouped operations
    resource_stats = df_slim.groupby("resource_id").agg({
        "task_id": "count",
        "case_id": "nunique"
    }).rename(columns={
        "task_id": "activity_count", 
        "case_id": "case_count"
    })
    
    # Calculate unique activities per resource with vectorized operations
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
    
    # Free memory
    if memory_efficient:
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
    if len(values) <= 1 or np.sum(values) == 0:
        return 0
    
    # Sort values (required for Lorenz curve)
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Calculate using optimized numpy operations for Lorenz curve
    # Faster than calculating point by point
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values))) - ((n + 1) / n)

def analyze_resource_performance(
    df: pd.DataFrame, 
    memory_efficient: bool = True
) -> pd.DataFrame:
    """
    Analyze resource performance metrics with memory optimization
    
    Args:
        df: Process data dataframe
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        DataFrame with resource performance metrics
    """
    start_time = time.time()
    logger.info("Analyzing resource performance...")
    
    if memory_efficient:
        # Select only necessary columns
        df_slim = df[['resource_id', 'task_id', 'timestamp', 'case_id']].copy()
    else:
        df_slim = df
    
    # Calculate resource performance metrics
    # For each resource, we'll compute:
    # 1. Average processing time per task
    # 2. Throughput (tasks per hour)
    
    # Add next timestamp within same case to calculate task duration
    df_slim = df_slim.sort_values(['case_id', 'timestamp'])
    df_slim['next_timestamp'] = df_slim.groupby('case_id')['timestamp'].shift(1)
    df_slim['task_duration'] = (df_slim['timestamp'] - df_slim['next_timestamp']).dt.total_seconds() / 3600
    
    # Filter out negative durations (from first task in case)
    df_slim = df_slim[df_slim['task_duration'] > 0]
    
    # Group by resource and calculate performance metrics
    resource_perf = df_slim.groupby('resource_id').agg({
        'task_duration': ['mean', 'median', 'std', 'count'],
        'task_id': 'count',
        'timestamp': [min, max]
    })
    
    # Flatten column names
    resource_perf.columns = ['_'.join(col).strip() for col in resource_perf.columns.values]
    
    # Calculate throughput (tasks per hour)
    resource_perf['total_hours'] = (
        resource_perf['timestamp_max'] - resource_perf['timestamp_min']
    ).dt.total_seconds() / 3600
    
    resource_perf['throughput'] = resource_perf['task_id_count'] / resource_perf['total_hours']
    
    # Sort by task count
    resource_perf = resource_perf.sort_values('task_id_count', ascending=False)
    
    # Log summary
    logger.info(f"Analyzed performance for {len(resource_perf)} resources")
    if len(resource_perf) > 0:
        avg_throughput = resource_perf['throughput'].mean()
        logger.info(f"Average throughput: {avg_throughput:.2f} tasks per hour")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    # Free memory
    if memory_efficient:
        del df_slim
        gc.collect()
    
    return resource_perf

def perform_conformance_checking(
    df: pd.DataFrame, 
    reference_variants: Optional[List] = None,
    memory_efficient: bool = True
) -> Dict[str, Any]:
    """
    Perform simplified conformance checking against reference variants
    
    Args:
        df: Process data dataframe
        reference_variants: Optional list of reference process variants
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        Dictionary with conformance metrics
    """
    start_time = time.time()
    logger.info("Performing conformance checking...")
    
    # Identify process variants
    variant_stats, variant_sequences = identify_process_variants(
        df, max_variants=100, memory_efficient=memory_efficient
    )
    
    # If reference variants not provided, use top variant as reference
    if not reference_variants and len(variant_sequences) > 0:
        reference_variants = [variant_sequences[1]]  # Use top variant
    
    if not reference_variants:
        logger.warning("No reference variants available for conformance checking")
        return {
            "conformance_ratio": 0,
            "variant_coverage": 0,
            "nonconforming_cases": []
        }
    
    # Calculate conformance metrics
    # Simply check for an exact match with any reference variant
    conforming_variants = set()
    for variant_id, sequence in variant_sequences.items():
        if any(np.array_equal(sequence, ref) for ref in reference_variants):
            conforming_variants.add(variant_id)
    
    # Count conforming cases
    conforming_cases = variant_stats[
        variant_stats['variant_id'].isin(conforming_variants)
    ]['count'].sum()
    
    total_cases = variant_stats['count'].sum()
    conformance_ratio = conforming_cases / total_cases if total_cases > 0 else 0
    
    # Get non-conforming variant IDs
    nonconforming_variants = [
        vid for vid in variant_sequences.keys() 
        if vid not in conforming_variants
    ]
    
    # Limit nonconforming variants to top ones for memory efficiency
    nonconforming_variants = nonconforming_variants[:min(len(nonconforming_variants), 20)]
    
    # Log summary
    logger.info(f"Conformance ratio: {conformance_ratio:.2%}")
    logger.info(f"Conforming cases: {conforming_cases} out of {total_cases}")
    logger.info(f"Analysis completed in {time.time() - start_time:.2f}s")
    
    return {
        "conformance_ratio": conformance_ratio,
        "variant_coverage": len(conforming_variants) / len(variant_sequences) if variant_sequences else 0,
        "conforming_variants": list(conforming_variants),
        "nonconforming_variants": nonconforming_variants
    }

def run_complete_analysis(
    df: pd.DataFrame, 
    memory_efficient: bool = True
) -> Dict[str, Any]:
    """
    Run all process mining analyses with optimized memory usage
    
    Args:
        df: Process data dataframe
        memory_efficient: Whether to use memory-efficient processing
        
    Returns:
        Dictionary with all analysis results
    """
    start_time = time.time()
    logger.info("Running complete process mining analysis...")
    
    # Run bottleneck analysis
    bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
        df, memory_efficient=memory_efficient
    )
    
    # Run cycle time analysis
    case_stats, long_cases, p95 = analyze_cycle_times(
        df, memory_efficient=memory_efficient
    )
    
    # Run transition pattern analysis
    transitions, trans_count, prob_matrix = analyze_transition_patterns(
        df, memory_efficient=memory_efficient
    )
    
    # Run variant analysis
    variant_stats, variant_sequences = identify_process_variants(
        df, memory_efficient=memory_efficient
    )
    
    # Run resource analysis
    resource_stats = analyze_resource_workload(
        df, memory_efficient=memory_efficient
    )
    
    # Compile metrics for summary
    metrics = {
        "cases": df["case_id"].nunique(),
        "events": len(df),
        "activities": df["task_id"].nunique(),
        "resources": df["resource_id"].nunique(),
        "variants": len(variant_stats),
        "bottlenecks": len(significant_bottlenecks),
        "perf": {
            "top_bottleneck_wait": significant_bottlenecks["mean_hours"].iloc[0] if len(significant_bottlenecks) > 0 else 0,
            "median_cycle_time": case_stats["duration_h"].median(),
            "p95_cycle_time": p95,
            "resource_gini": resource_stats.attrs.get("gini_coefficient", 0),
            "top_variant_pct": variant_stats["percentage"].iloc[0] if len(variant_stats) > 0 else 0
        }
    }
    
    # Calculate overall process score (0-100)
    # This is a simple scoring formula that can be customized
    # Higher score means better process
    score_components = [
        100 - min(100, metrics["perf"]["top_bottleneck_wait"] * 5),  # Low bottleneck wait is good
        100 - min(100, metrics["perf"]["p95_cycle_time"] / 24 * 10),  # Low cycle time is good
        100 - min(100, metrics["perf"]["resource_gini"] * 100),  # Low Gini (balanced workload) is good
        min(100, metrics["perf"]["top_variant_pct"])  # High variant conformance is good
    ]
    
    # Overall score is average of components
    metrics["process_score"] = sum(score_components) / len(score_components)
    
    # Calculate total analysis time
    total_time = time.time() - start_time
    logger.info(f"Complete analysis completed in {total_time:.2f}s")
    
    return {
        "bottleneck_stats": bottleneck_stats,
        "significant_bottlenecks": significant_bottlenecks,
        "case_stats": case_stats,
        "long_cases": long_cases,
        "transitions": transitions,
        "trans_count": trans_count,
        "prob_matrix": prob_matrix,
        "variant_stats": variant_stats,
        "variant_sequences": variant_sequences,
        "resource_stats": resource_stats,
        "metrics": metrics,
        "analysis_time": total_time
    }